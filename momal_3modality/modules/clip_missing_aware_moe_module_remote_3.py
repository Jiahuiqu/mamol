import torch
import torch.nn as nn
import pytorch_lightning as pl
import LMMOE_3modality.modules.vision_transformer_moe_3 as vit
from LMMOE.modules import clip_utils, objectives
import copy


# ============================================================
# CLIP backbone loader
# ============================================================
def load_clip_to_cpu():
    model_path = 'ViT-B-16.pt'
    try:
        model = torch.jit.load(model_path, map_location="cpu")
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = vit.build_model(state_dict or model.state_dict())
    return model


# ============================================================
# Embedding Layer: HS + LiDAR + SAR
# ============================================================
class EmbeddingLayer3(nn.Module):
    def __init__(self, in_ch_hs=180, in_ch_lidar=1, in_ch_sar=4):
        super().__init__()

        # HS
        self.conv_HS = nn.Conv2d(in_ch_hs, 256, 3, 1, 1, bias=False)
        self.conv_HS_2 = nn.Conv2d(256, 768, 1, 1, bias=False)

        # LiDAR
        self.conv_LIDAR = nn.Conv2d(in_ch_lidar, 256, 3, 1, 1, bias=False)
        self.conv_LIDAR_2 = nn.Conv2d(256, 768, 1, 1, bias=False)

        # SAR
        self.conv_SAR = nn.Conv2d(in_ch_sar, 256, 3, 1, 1, bias=False)
        self.conv_SAR_2 = nn.Conv2d(256, 768, 1, 1, bias=False)

    def forward(self, HS, LIDAR, SAR):
        def encode(x, conv1, conv2):
            x = conv1(x)
            x = conv2(x)
            return x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

        hs_feat = encode(HS, self.conv_HS, self.conv_HS_2)
        lidar_feat = encode(LIDAR, self.conv_LIDAR, self.conv_LIDAR_2)
        sar_feat = encode(SAR, self.conv_SAR, self.conv_SAR_2)
        return hs_feat, lidar_feat, sar_feat


# ============================================================
# Custom CLIP backbone wrapper (3-modal)
# ============================================================
class CustomCLIP3(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.dtype = clip_model.dtype

    def forward(self, image1, image2, image3, missing_type):
        feat1 = self.image_encoder(image1.type(self.dtype), missing_type)
        feat2 = self.image_encoder(image2.type(self.dtype), missing_type)
        feat3 = self.image_encoder(image3.type(self.dtype), missing_type)
        return torch.cat([feat1, feat2, feat3], dim=-1)


# ============================================================
# Main LightningModule
# ============================================================
class CLIPransformerSS_3(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        clip_model = load_clip_to_cpu()

        hsin = config["hs_in"]
        lidarin = config["lidar_in"]
        sarin = config["sar_in"]

        self.embedding_layer = EmbeddingLayer3(hsin, lidarin, sarin)
        print("Building 3-modality CLIP")

        hidden_size = 512 * 3  
        self.model = CustomCLIP3(clip_model)

        # ===================== Downstream ===================== #
        if config["loss_names"]["remote"] > 0:
            cls_num = config["remote_class_num"]
            self.remote_classifier = nn.Linear(hidden_size, cls_num)
            self.remote_classifier.apply(objectives.init_weights)

        # 默认冻结 backbone
        for _, p in self.model.named_parameters():
            p.requires_grad = False

        for n, p in self.model.named_parameters():
            if "mamol" in n or "missing_proj" in n:
                p.requires_grad = True

        clip_utils.set_metrics(self)
        self.current_tasks = list()
        self.records = {}

    # ============================================================
    # Inference
    # ============================================================
    def infer(self, batch):
        HS = batch["HS"]
        LIDAR = batch["lidar"]
        SAR = batch["SAR"]

        hs_feat, lidar_feat, sar_feat = self.embedding_layer(HS, LIDAR, SAR)
        all_feats = self.model(hs_feat, lidar_feat, sar_feat, batch["missing_type"])

        feat_dim = all_feats.shape[1] // 3

        for i in range(len(HS)):
            mt = batch["missing_type"][i]
            if mt == 0:
                continue
            # 1: HS, 2: Lidar, 4: SAR
            if mt & 1:  #  HS
                all_feats[i, :feat_dim].zero_()
            if mt & 2:  #  LiDAR
                all_feats[i, feat_dim:2*feat_dim].zero_()
            if mt & 4:  #  SAR
                all_feats[i, 2*feat_dim:].zero_()

        return {"cls_feats": all_feats}

    # ============================================================
    # Forward
    # ============================================================
    def forward(self, batch):
        ret = {}
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret
        if "remote" in self.current_tasks:
            ret.update(objectives.compute_remote(self, batch))
        return ret

    # ============================================================
    # Training / Validation / Testing
    # ============================================================
    def training_step(self, batch, batch_idx):
        clip_utils.set_task(self)
        output = self(batch)
        total_loss = sum(v for k, v in output.items() if "loss" in k)
        return total_loss

    def training_epoch_end(self, outs):
        clip_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        clip_utils.set_task(self)
        _ = self(batch)

    def validation_epoch_end(self, outs):
        clip_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        clip_utils.set_task(self)
        _ = self(batch)

    def test_epoch_end(self, outs):
        clip_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return clip_utils.set_schedule(self)
