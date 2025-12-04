import torch
import torch.nn as nn
import pytorch_lightning as pl
import LMMOE.modules.vision_transformer_prompts as vit
import math
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from LMMOE.modules import clip_utils, heads, objectives, clip
import copy

def load_clip_to_cpu(backbone_name, prompt_length, prompt_depth):
    url = clip._MODELS[backbone_name]
    # model_path = clip._download(url)
    model_path = 'ViT-B-16.pt'

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu")#.eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = vit.build_model(state_dict or model.state_dict(), prompt_length, prompt_depth)

    return model

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.prompt_length = clip_model.prompt_length

    def forward(self, tokenized_texts, all_prompts_text, missing_type):
        x = self.token_embedding(tokenized_texts).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, all_prompts_text, 0, missing_type]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0][self.prompt_length:]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_texts.argmax(dim=-1)] @ self.text_projection

        return x

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class MultiModalPromptLearner(nn.Module):
    def __init__(self, prompt_length, prompt_depth, clip_model):
        super().__init__()
        dtype = clip_model.dtype
        prompt_length_half = prompt_length//3 # use half length for generating static prompts, and the other for generating dynamic prompts
        # Default is 1, which is compound shallow prompting
        self.prompt_depth = prompt_depth  # max=12, but will create 11 such shared prompts
        self.visual_prompt_complete = nn.Parameter(nn.init.normal_(torch.empty(prompt_length_half, 768, dtype=dtype), std=0.02))
        self.visual_prompt_missing = nn.Parameter(nn.init.normal_(torch.empty(prompt_length_half, 768, dtype=dtype), std=0.02))
        self.text_prompt_complete = nn.Parameter(nn.init.normal_(torch.empty(prompt_length_half, 512, dtype=dtype), std=0.02))
        self.text_prompt_missing = nn.Parameter(nn.init.normal_(torch.empty(prompt_length_half, 512, dtype=dtype), std=0.02))
        self.common_prompt_complete = nn.Parameter(nn.init.normal_(torch.empty(prompt_length_half, 512, dtype=dtype), std=0.02))
        self.common_prompt_image = nn.Parameter(nn.init.normal_(torch.empty(prompt_length_half, 512, dtype=dtype), std=0.02))
        self.common_prompt_text = nn.Parameter(nn.init.normal_(torch.empty(prompt_length_half, 512, dtype=dtype), std=0.02))
        # Also make corresponding projection layers, for each prompt
        embed_dim_text = 512
        embed_dim_image = 768
        embed_dim = embed_dim_text + embed_dim_image
        r = 16
        single_layer = nn.Sequential(
                nn.Linear(embed_dim, embed_dim//r),
                nn.GELU(),
                nn.Linear(embed_dim//r, embed_dim_text),
                )
        self.compound_prompt_projections_text = _get_clones(single_layer, self.prompt_depth)
        self.layernorm_text = nn.ModuleList([torch.nn.LayerNorm(embed_dim) for _ in range(self.prompt_depth)])
        
        single_layer = nn.Sequential(
                nn.Linear(embed_dim, embed_dim//r),
                nn.GELU(),
                nn.Linear(embed_dim//r, embed_dim_image),
                )
        self.compound_prompt_projections_image = _get_clones(single_layer, self.prompt_depth)
        self.layernorm_image = nn.ModuleList([torch.nn.LayerNorm(embed_dim) for _ in range(self.prompt_depth)])
        self.common_prompt_projection_image = nn.Sequential(
                nn.Linear(embed_dim_text, embed_dim_text//r),
                nn.GELU(),
                nn.Linear(embed_dim_text//r, embed_dim_image),
                )
        self.common_prompt_projection_text = nn.Sequential(
                nn.Linear(embed_dim_text, embed_dim_text//r),
                nn.GELU(),
                nn.Linear(embed_dim_text//r, embed_dim_text),
                )

    def forward(self, missing_type):

        # Before returning, need to transform
        # prompts to 768 for the visual side
        all_prompts_image = [ [] for _ in range(self.prompt_depth)]   # Prompts of prompt_depth layers
        all_prompts_text = [ [] for _ in range(self.prompt_depth)]   # Prompts of prompt_depth layers
        for i in range(len(missing_type)):
            # set initial prompts for each modality
            if missing_type[i]==0:  # modality complete
                initial_prompt_image = self.visual_prompt_complete
                initial_prompt_text = self.text_prompt_complete
                common_prompt = self.common_prompt_complete
            elif missing_type[i]==1:  # missing text 
                initial_prompt_image = self.visual_prompt_complete
                initial_prompt_text = self.text_prompt_missing
                common_prompt = self.common_prompt_image
            elif missing_type[i]==2:  # missing image 
                initial_prompt_image = self.visual_prompt_missing
                initial_prompt_text = self.text_prompt_complete
                common_prompt = self.common_prompt_text
            # generate the prompts of the first layer
            all_prompts_image[0].append(self.compound_prompt_projections_image[0](self.layernorm_image[0](torch.cat([initial_prompt_image, initial_prompt_text], -1))))
            all_prompts_text[0].append(self.compound_prompt_projections_text[0](self.layernorm_text[0](torch.cat([initial_prompt_image, initial_prompt_text], -1))))
            # generate the prompts of the rest layers
            for index in range(1, self.prompt_depth):
                all_prompts_image[index].append(
                    self.compound_prompt_projections_image[index](self.layernorm_image[index](torch.cat([all_prompts_image[index-1][-1], all_prompts_text[index-1][-1]], -1))))
                all_prompts_text[index].append(
                    self.compound_prompt_projections_text[index](self.layernorm_text[index](torch.cat([all_prompts_image[index-1][-1], all_prompts_text[index-1][-1]], -1))))
            all_prompts_image[0][i] = torch.cat([
                    all_prompts_image[0][i], 
                    self.common_prompt_projection_image(common_prompt)]
                    ,0)
            all_prompts_text[0][i] = torch.cat([
                    all_prompts_text[0][i], 
                    self.common_prompt_projection_text(common_prompt)]
                    ,0)
        # generate the prompts in each layer as a tensor [B, L, C]
        all_prompts_image = [torch.stack(prompts) for prompts in all_prompts_image]
        all_prompts_text = [torch.stack(prompts) for prompts in all_prompts_text]
        return all_prompts_image, all_prompts_text   

class CustomCLIP(nn.Module):
    def __init__(self, prompt_length, prompt_depth, clip_model):
        super().__init__()
        self.prompt_learner = MultiModalPromptLearner(prompt_length, prompt_depth, clip_model)
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, text, missing_type):
        tokenized_texts = torch.stack([clip.tokenize(tx, context_length=77, truncate=True) for tx in text[0]], 0).to(image.get_device()).squeeze(1)  # extract texts from the first key  # [b, 77]
        #logit_scale = self.logit_scale.exp()

        all_prompts_image, all_prompts_text = self.prompt_learner(missing_type)
        text_features = self.text_encoder(tokenized_texts, all_prompts_text, missing_type)
        image_features = self.image_encoder(image.type(self.dtype), all_prompts_image, missing_type)
        return torch.cat([image_features, text_features], -1)

class CLIPransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        clip_model = load_clip_to_cpu(config['vit'], config['prompt_length'], config['prompt_depth'])

        print("Building custom CLIP")
        hidden_size = 512*2
        self.model = CustomCLIP(config['prompt_length'], config['prompt_depth'], clip_model)

        # ===================== Downstream ===================== #
        if (
            self.hparams.config["load_path"] != ""
            and not self.hparams.config["test_only"]
            and not self.hparams.config["finetune_first"]
        ):
# 
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.model.load_state_dict(state_dict, strict=False)

        if self.hparams.config["loss_names"]["hatememes"] > 0:
            cls_num = self.hparams.config["hatememes_class_num"]
            self.hatememes_classifier = nn.Linear(hidden_size, cls_num)
            self.hatememes_classifier.apply(objectives.init_weights)
            
        if self.hparams.config["loss_names"]["food101"] > 0:
            cls_num = self.hparams.config["food101_class_num"]
            self.food101_classifier = nn.Linear(hidden_size, cls_num)
            self.food101_classifier.apply(objectives.init_weights)               
            
        if self.hparams.config["loss_names"]["mmimdb"] > 0:
            cls_num = self.hparams.config["mmimdb_class_num"]
            self.mmimdb_classifier = nn.Linear(hidden_size, cls_num)
            self.mmimdb_classifier.apply(objectives.init_weights)  
            
        if self.hparams.config["load_path"] != "" and self.hparams.config["finetune_first"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.model.load_state_dict(state_dict, strict=False)            
            print("use pre-finetune model")

        if not self.hparams.config["test_only"]:
            for name, param in self.model.named_parameters():
                if "prompt_learner" not in name and "prompt" not in name and 'ln_final' not in name and 'ln_post' not in name and name.split('.')[-1]!='proj':
                    param.requires_grad_(False)

            # # Double check
            # enabled = set()
            # for name, param in self.model.named_parameters():
            #     if param.requires_grad:
            #         enabled.add(name)
            # print(f"Parameters to be updated: {enabled}")

        clip_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=True)
        self.records = {}

    def debug_batch(self, batch):
        print("\n=== Batch Debug Info ===")
        for k, v in batch.items():
            if hasattr(v, "shape"):  # tensor
                print(f"{k}: {type(v)}, shape={v.shape}, dtype={v.dtype}")
            elif isinstance(v, (list, tuple)):
                print(f"{k}: {type(v)}, len={len(v)}")
                if len(v) > 0 and hasattr(v[0], "shape"):
                    print(f"   first element shape={v[0].shape}, dtype={v[0].dtype}")
            else:
                print(f"{k}: {type(v)}, value={v}")
        print("========================\n")

    def infer(
        self,
        batch,
    ):
        # self.debug_batch(batch)
        text = batch["text"]
        img = batch["image"][0]  # extract the first view (total 1)
        if self.hparams.config["test_only"]:
            self.model.eval()
            if self.hparams.config["loss_names"]["hatememes"] > 0:
                self.hatememes_classifier.eval()
            
            if self.hparams.config["loss_names"]["food101"] > 0:
                self.food101_classifier.eval()            
            
            if self.hparams.config["loss_names"]["mmimdb"] > 0:
                self.mmimdb_classifier.eval()
        both_feats = self.model(img, text, batch["missing_type"])  
        feature_dim = both_feats.shape[1]//2
        for idx in range(len(img)):
            if batch["missing_type"][idx] == 0:
                pass       
            elif batch["missing_type"][idx] == 1:  # missing text
                both_feats[idx, feature_dim:].zero_()
            elif batch["missing_type"][idx] == 2:
                both_feats[idx, :feature_dim].zero_()
            
        ret = {
            "cls_feats": both_feats,
        }

        return ret

    def forward(self, batch):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        # Masked Patch Prediction
        if "mpp" in self.current_tasks:
            ret.update(objectives.compute_mpp(self, batch))

        # Image Text Matching
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm_wpa(self, batch))
            
        # Binary classification for Hateful Memes
        if "hatememes" in self.current_tasks:
            ret.update(objectives.compute_hatememes(self, batch))
            
        # Multi-label classification for MM-IMDb
        if "mmimdb" in self.current_tasks:
            ret.update(objectives.compute_mmimdb(self, batch))
            
        # Classification for Food101
        if "food101" in self.current_tasks:
            ret.update(objectives.compute_food101(self, batch))              

        return ret

    def training_step(self, batch, batch_idx):
        clip_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outs):
        clip_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        clip_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        clip_utils.epoch_wrapup(self)
#         print('missing_img:', self.missing_img_prompt[0,0:3,0:8])
#         print('missing_text:', self.missing_text_prompt[0,0:3,0:8])
#         print('complete:', self.complete_prompt[0,0:3,0:8])

    def test_step(self, batch, batch_idx):
        clip_utils.set_task(self)
        output = self(batch)
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))

        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name)
        clip_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return clip_utils.set_schedule(self)
