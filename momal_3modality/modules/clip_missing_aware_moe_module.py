import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from LMMOE.modules import clip_utils, objectives, clip
# import LMMOE.modules.vision_transformer_moe as vit_moe
import LMMOE.modules.vision_transformer_moe as vit_moe

import torch.nn.functional as F
#

def load_clip_to_cpu(backbone_name):
    url = clip._MODELS[backbone_name]
    # model_path = clip._download(url)
    model_path = 'ViT-B-16.pt'

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu")#.eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = vit_moe.build_model(state_dict or model.state_dict())
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

    def forward(self, tokenized_texts, missing_type):
        # tokenized_texts: [B, 77]
        x = self.token_embedding(tokenized_texts).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # [L, B, D]
        x = self.transformer(x, missing_type)
        x = x.permute(1, 0, 2)  # [B, L, D]
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_texts.argmax(dim=-1)] @ self.text_projection
        return x  # [B, 512]

class CustomCLIP(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual  
        self.text_encoder = TextEncoder(clip_model)
        self.dtype = clip_model.dtype

    def forward(self, image, text, missing_type):
        # 1. encode text
        # tokenized_texts = torch.stack([clip.tokenize(tx, context_length=77, truncate=True) for tx in text[0]], 0).to(image.get_device()).squeeze(1)  # extract texts from the first key  # [b, 77]
        tokenized_texts = torch.stack(
            [clip.tokenize(tx, context_length=77, truncate=True) for tx in text[0]], 0
        ).to(image.device).squeeze(1)

        text_features = self.text_encoder(tokenized_texts, missing_type)  

        # 2. encode image
        image_features = self.image_encoder(image.type(self.dtype), missing_type)

        # 3. cat
        feats = torch.cat([image_features, text_features], dim=-1)

        return feats  # [B, 1024]


class CLIPransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        clip_model = load_clip_to_cpu(config['vit'])
        print("Building custom CLIP")
        hidden_size = 512 * 2
        self.model = CustomCLIP(clip_model)

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
                if "prompt_learner" not in name and "prompt" not in name and 'ln_final' not in name and 'ln_post' not in name and \
                        name.split('.')[-1] != 'proj':
                    param.requires_grad_(False)
                # param.data = param.data.to(torch.float32)
            # for n, p in self.model.named_parameters():
            #     p.requires_grad = False
            for n, p in self.model.named_parameters():
                if "mamol" in n or "missing_proj" in n:
                    p.requires_grad = True
            # for name, param in self.model.named_parameters():
            #     if param.requires_grad:
            #         print(name)

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
        feature_dim = both_feats.shape[1] // 2
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

