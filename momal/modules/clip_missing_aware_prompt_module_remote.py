import torch
import torch.nn as nn
import pytorch_lightning as pl
import LMMOE.modules.vision_transformer_prompts_remote as vit
import math
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from LMMOE.modules import clip_utils, heads, objectives, clip
import copy

def load_clip_to_cpu(backbone_name, prompt_length, prompt_depth):
    # url = clip._MODELS[backbone_name]
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

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
class DualVisualPromptLearner(nn.Module):
    def __init__(self, prompt_length, prompt_depth, clip_model):
        super().__init__()
        dtype = clip_model.dtype
        prompt_length_half = prompt_length // 3
        self.prompt_depth = prompt_depth  

        self.visual1_prompt_complete = nn.Parameter(
            nn.init.normal_(torch.empty(prompt_length_half, 768, dtype=dtype), std=0.02))
        self.visual1_prompt_missing = nn.Parameter(
            nn.init.normal_(torch.empty(prompt_length_half, 768, dtype=dtype), std=0.02))

        self.visual2_prompt_complete = nn.Parameter(
            nn.init.normal_(torch.empty(prompt_length_half, 768, dtype=dtype), std=0.02))
        self.visual2_prompt_missing = nn.Parameter(
            nn.init.normal_(torch.empty(prompt_length_half, 768, dtype=dtype), std=0.02))

        self.common_prompt_complete = nn.Parameter(
            nn.init.normal_(torch.empty(prompt_length_half, 768, dtype=dtype), std=0.02))
        self.common_prompt_image1 = nn.Parameter(
            nn.init.normal_(torch.empty(prompt_length_half, 768, dtype=dtype), std=0.02))
        self.common_prompt_image2 = nn.Parameter(
            nn.init.normal_(torch.empty(prompt_length_half, 768, dtype=dtype), std=0.02))

        embed_dim_image1 = 768
        embed_dim_image2 = 768
        embed_dim = embed_dim_image1 + embed_dim_image2
        r = 16

        single_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // r),
            nn.GELU(),
            nn.Linear(embed_dim // r, embed_dim_image2),
        )
        self.compound_prompt_projections_image2 = _get_clones(single_layer, self.prompt_depth)
        self.layernorm_image2 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(self.prompt_depth)])

        single_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // r),
            nn.GELU(),
            nn.Linear(embed_dim // r, embed_dim_image1),
        )
        self.compound_prompt_projections_image1 = _get_clones(single_layer, self.prompt_depth)
        self.layernorm_image1 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(self.prompt_depth)])

        self.common_prompt_projection_image1 = nn.Sequential(
            nn.Linear(embed_dim_image2, embed_dim_image2 // r),
            nn.GELU(),
            nn.Linear(embed_dim_image2 // r, embed_dim_image1),
        )
        self.common_prompt_projection_image2 = nn.Sequential(
            nn.Linear(embed_dim_image2, embed_dim_image2 // r),
            nn.GELU(),
            nn.Linear(embed_dim_image2 // r, embed_dim_image2),
        )

    def forward(self, missing_type):
 
        all_prompts_image1 = [[] for _ in range(self.prompt_depth)]
        all_prompts_image2 = [[] for _ in range(self.prompt_depth)]

        for i in range(len(missing_type)):
            if missing_type[i] == 0:  
                initial_prompt_image1 = self.visual1_prompt_complete
                initial_prompt_image2 = self.visual2_prompt_complete
                common_prompt = self.common_prompt_complete
            elif missing_type[i] == 1:  
                initial_prompt_image1 = self.visual1_prompt_complete
                initial_prompt_image2 = self.visual2_prompt_missing
                common_prompt = self.common_prompt_image1
            elif missing_type[i] == 2:  
                initial_prompt_image1 = self.visual1_prompt_missing
                initial_prompt_image2 = self.visual2_prompt_complete
                common_prompt = self.common_prompt_image2
            else:
                raise ValueError(f"Invalid missing_type {missing_type[i]}")

            concat_feat = torch.cat([initial_prompt_image1, initial_prompt_image2], dim=-1)
            all_prompts_image1[0].append(
                self.compound_prompt_projections_image1[0](self.layernorm_image1[0](concat_feat)))
            all_prompts_image2[0].append(
                self.compound_prompt_projections_image2[0](self.layernorm_image2[0](concat_feat)))

            for index in range(1, self.prompt_depth):
                concat_feat = torch.cat(
                    [all_prompts_image1[index - 1][-1], all_prompts_image2[index - 1][-1]], dim=-1)
                all_prompts_image1[index].append(
                    self.compound_prompt_projections_image1[index](self.layernorm_image1[index](concat_feat)))
                all_prompts_image2[index].append(
                    self.compound_prompt_projections_image2[index](self.layernorm_image2[index](concat_feat)))

            all_prompts_image1[0][i] = torch.cat([
                all_prompts_image1[0][i],
                self.common_prompt_projection_image1(common_prompt)
            ], dim=0)
            all_prompts_image2[0][i] = torch.cat([
                all_prompts_image2[0][i],
                self.common_prompt_projection_image2(common_prompt)
            ], dim=0)

        all_prompts_image1 = [torch.stack(p) for p in all_prompts_image1]
        all_prompts_image2 = [torch.stack(p) for p in all_prompts_image2]
        return all_prompts_image1, all_prompts_image2

class CustomCLIP2(nn.Module):
    def __init__(self, prompt_length, prompt_depth, clip_model1, clip_model2):
        super().__init__()
        self.prompt_learner = DualVisualPromptLearner(prompt_length, prompt_depth, clip_model1)

        self.image_encoder = clip_model1.visual
        self.dtype = clip_model1.dtype
        self.text_encoder = clip_model2.visual

    def forward(self, image1, image2, missing_type):
        missing_type = missing_type
        all_prompts_image1, all_prompts_image2 = self.prompt_learner(missing_type)
        # print(image1.shape)
        image_features1 = self.image_encoder(image1.type(self.dtype), all_prompts_image1, missing_type)
        image_features2 = self.text_encoder(image2.type(self.dtype), all_prompts_image2, missing_type)
        return torch.cat([image_features1, image_features2], -1)
class CustomCLIP1(nn.Module):
    def __init__(self, prompt_length, prompt_depth, clip_model):
        super().__init__()
        self.prompt_learner = DualVisualPromptLearner(prompt_length, prompt_depth, clip_model)

        self.image_encoder = clip_model.visual
        self.dtype = clip_model.dtype

    def forward(self, image1, image2, missing_type):
        missing_type = missing_type
        all_prompts_image1, all_prompts_image2 = self.prompt_learner(missing_type)
        # print(image1.shape)
        image_features1 = self.image_encoder(image1.type(self.dtype), all_prompts_image1, missing_type)
        image_features2 = self.image_encoder(image2.type(self.dtype), all_prompts_image2, missing_type)
        # print("image1 mean:", image1.abs().mean().item(), "image2 mean:", image2.abs().mean().item())
        return torch.cat([image_features1, image_features2], -1)

class CustomCLIP3(nn.Module):
    def __init__(self, prompt_length, prompt_depth, clip_model1, clip_model2):
        super().__init__()
        self.prompt_learner = DualVisualPromptLearner(prompt_length, prompt_depth, clip_model1)

        self.image_encoder1 = clip_model1.visual
        self.image_encoder2 = clip_model2.visual
        self.dtype = clip_model1.dtype

    def forward(self, image1, image2, missing_type):
        missing_type = missing_type
        all_prompts_image1, all_prompts_image2 = self.prompt_learner(missing_type)
        # print(image1.shape)
        image_features1 = self.image_encoder1(image1.type(self.dtype), all_prompts_image1, missing_type)
        image_features2 = self.image_encoder2(image2.type(self.dtype), all_prompts_image2, missing_type)
        # print("image1 mean:", image1.abs().mean().item(), "image2 mean:", image2.abs().mean().item())
        image_features = torch.cat([image_features1, image_features2], -1)
        # return torch.cat([image_features1, image_features2], -1)
        # print(image_features)
        return image_features

class embedding_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_HS = nn.Conv2d(in_channels=63, out_channels=256,
                                 kernel_size=3, stride=1, padding=1 ,bias=False)
        self.conv_LIDAR = nn.Conv2d(in_channels=1, out_channels=256,
                                    kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_HS_2 = nn.Conv2d(in_channels=256, out_channels=768,
                                 kernel_size=1, stride=1, bias=False)
        self.conv_LIDAR_2 = nn.Conv2d(in_channels=256, out_channels=768,
                                    kernel_size=1, stride=1, bias=False)
    def forward(self, HS, LIDAR):
        HS = self.conv_HS(HS)
        LIDAR = self.conv_LIDAR(LIDAR)
        HS = self.conv_HS_2(HS)
        LIDAR = self.conv_LIDAR_2(LIDAR)
        HS = HS.reshape(HS.shape[0], HS.shape[1], -1).permute(0, 2, 1)
        LIDAR = LIDAR.reshape(LIDAR.shape[0], LIDAR.shape[1], -1).permute(0, 2, 1)

        return HS, LIDAR

class CLIPransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        clip_model = load_clip_to_cpu(config['vit'], config['prompt_length'], config['prompt_depth'])

        # clip_model1 = load_clip_to_cpu(config['vit'], config['prompt_length'], config['prompt_depth'])
        # clip_model2 = load_clip_to_cpu(config['vit'], config['prompt_length'], config['prompt_depth'])
        self.embedding_layer = embedding_layer()
        print("Building custom CLIP")
        hidden_size = 512*2
        self.model = CustomCLIP1(config['prompt_length'], config['prompt_depth'], clip_model)
        # self.model = CustomCLIP3(config['prompt_length'], config['prompt_depth'], clip_model1, clip_model2)


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


        if self.hparams.config["loss_names"]["remote"] > 0:
            cls_num = self.hparams.config["remote_class_num"]
            self.remote_classifier = nn.Linear(hidden_size, cls_num)
            self.remote_classifier.apply(objectives.init_weights)

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

    def infer(
        self,
        batch,
    ):
        HS = batch["HS"]
        LIDAR = batch["lidar"]
        # print("HS shape:", HS)
        # print("LIDAR shape:", LIDAR)

        # image1 = self.conv_HS(HS)
        # image1 = image1.reshape(image1.shape[0], image1.shape[1], -1).permute(0, 2, 1)
        # image2 = self.conv_LIDAR(LIDAR)
        # image2 = image2.reshape(image2.shape[0], image2.shape[1], -1).permute(0, 2, 1)

        image1, image2 = self.embedding_layer(HS, LIDAR)
        # print("img2:", image1)
        both_feats = self.model(image1, image2, batch["missing_type"])
        feature_dim = both_feats.shape[1]//2
        for idx in range(len(HS)):
            if batch["missing_type"][idx] == 0:
                pass
            elif batch["missing_type"][idx] == 2:  # missing text
                both_feats[idx, feature_dim:].zero_()
            elif batch["missing_type"][idx] == 1:
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
        if "remote" in self.current_tasks:
            ret.update(objectives.compute_remote(self, batch))
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
        # print("current tasks:", self.current_tasks)
        output = self(batch)

        # self.on_after_backward()
        # mt = batch['missing_type']
        # print("missing_type unique/count:", mt)

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