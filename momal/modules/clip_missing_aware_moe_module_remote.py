import torch
import torch.nn as nn
import pytorch_lightning as pl
import LMMOE.modules.vision_transformer_moe_remote as vit
from LMMOE.modules import clip_utils, heads, objectives, clip
import copy

def load_clip_to_cpu():
    # url = clip._MODELS[backbone_name]
    # model_path = clip._download(url)
    model_path = 'ViT-B-16.pt'

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu")#.eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = vit.build_model(state_dict or model.state_dict())

    return model

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class CustomCLIP1(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.dtype = clip_model.dtype

    def forward(self, image1, image2, missing_type):
        image_features1 = self.image_encoder(image1.type(self.dtype), missing_type)
        image_features2 = self.image_encoder(image2.type(self.dtype), missing_type)
        return torch.cat([image_features1, image_features2], -1)

class embedding_layer(nn.Module):
    def __init__(self, in_ch_hs=144, in_ch_lidar=1):
        super().__init__()
        self.conv_HS = nn.Conv2d(in_channels=in_ch_hs, out_channels=256,
                                 kernel_size=3, stride=1, padding=1 ,bias=False)
        self.conv_LIDAR = nn.Conv2d(in_channels=in_ch_lidar, out_channels=256,
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
        clip_model = load_clip_to_cpu()
        hsin = config["hs_in"]
        lidarin = config["lidar_in"]
        self.embedding_layer = embedding_layer(hsin, lidarin)
        print("Building custom CLIP")
        hidden_size = 512*2
        self.model = CustomCLIP1(clip_model)

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

        if self.hparams.config["load_path"] != "" and self.hparams.config["finetune_first"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.model.load_state_dict(state_dict, strict=False)            
            print("use pre-finetune model")

        if not self.hparams.config["test_only"]:
            # for name, param in self.model.named_parameters():
            #     if "prompt_learner" not in name and "prompt" not in name and 'ln_final' not in name and 'ln_post' not in name and name.split('.')[-1]!='proj':
            #         param.requires_grad_(False)
            for _, p in self.model.named_parameters():
                p.requires_grad = False

            for n, p in self.model.named_parameters():
                if "mamol" in n or "missing_proj" in n:
                    p.requires_grad = True
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    print(name)
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
        image1, image2 = self.embedding_layer(HS, LIDAR)
        both_feats = self.model(image1, image2, batch["missing_type"])
        feature_dim = both_feats.shape[1]//2
        for idx in range(len(HS)):
            if batch["missing_type"][idx] == 0:
                pass
            elif batch["missing_type"][idx] == 2:
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

        return ret

    def training_step(self, batch, batch_idx):
        clip_utils.set_task(self)
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
