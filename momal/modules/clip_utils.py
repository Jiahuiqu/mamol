import torch
import random

from transformers.optimization import AdamW
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from LMMOE.modules.dist_utils import all_gather
from LMMOE.modules.objectives import compute_irtr_recall
from LMMOE.gadgets.my_metrics import Accuracy, VQAScore, Scalar, F1_Score, AUROC, Scalar2, check,ClassificationMetrics


def set_metrics(pl_module):
    for split in ["train", "val"]:
        for k, v in pl_module.hparams.config["loss_names"].items():
            if v < 1:
                continue
            if k == "vqa":
                setattr(pl_module, f"{split}_vqa_score", VQAScore())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            elif k == "remote":
                cls_num = pl_module.hparams.config.get("remote_class_num", 15) 
                setattr(pl_module, f"{split}_{k}_class_metrics", ClassificationMetrics(num_classes=cls_num))
                setattr(pl_module, f"{split}_{k}_loss", Scalar())

            elif k == "mmimdb":
                setattr(pl_module, f"{split}_{k}_F1_scores", F1_Score())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
                
            elif k == "hatememes":
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_AUROC", AUROC())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
                
            elif k == "food101":
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())       
                
            elif k == "nlvr2":
                if split == "train":
                    setattr(pl_module, f"train_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"train_{k}_loss", Scalar())
                else:
                    setattr(pl_module, f"dev_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"dev_{k}_loss", Scalar())
                    setattr(pl_module, f"test_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"test_{k}_loss", Scalar())
            elif k == "irtr":
                setattr(pl_module, f"{split}_irtr_loss", Scalar())
            elif k == "mppd" or k == "mpfr":
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            elif k == "itm":
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
                setattr(pl_module, f"{split}_{k}_wpa_loss", Scalar())
            else:
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())

def test_ablation(pl_module, loss_name, res):
    test_ratio = pl_module.hparams.config['test_ratio']
    exp_name = pl_module.hparams.config["test_exp_name"]
    test_type = pl_module.hparams.config["test_type"]       
    records = f'missing ratio: {test_ratio}, ' + res
    record_file = f'./records/{loss_name}/{loss_name}_{exp_name}_on_missing_{test_type}'
    with open(record_file, 'a+') as f:
        f.write(records+'\n')
                
def epoch_wrapup(pl_module):
    phase = "train" if pl_module.training else "val"
    the_metric = 0

    if pl_module.hparams.config["get_recall_metric"] and not pl_module.training:
        (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10) = compute_irtr_recall(pl_module)
        print((ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10), pl_module.global_step)
        pl_module.logger.experiment.add_scalar(
            "recalls/ir_r1", ir_r1, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/ir_r5", ir_r5, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/ir_r10", ir_r10, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/tr_r1", tr_r1, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/tr_r5", tr_r5, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/tr_r10", tr_r10, pl_module.global_step
        )
        the_metric += ir_r1.item() + tr_r1.item()

    for loss_name, v in pl_module.hparams.config["loss_names"].items():
        if v < 1:
            continue

        value = 0

        if loss_name == "vqa":
            value = getattr(pl_module, f"{phase}_{loss_name}_score").compute()
            pl_module.log(f"{loss_name}/{phase}/score_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_score").reset()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()



        elif loss_name == "remote":

            class_metrics = getattr(pl_module, f"{phase}_{loss_name}_class_metrics").compute()
            OA = class_metrics["OA"]
            AA = class_metrics["AA"]
            Kappa = class_metrics["Kappa"]
            # log到TensorBoard
            pl_module.log(f"{loss_name}/{phase}/OA_epoch", OA)
            pl_module.log(f"{loss_name}/{phase}/AA_epoch", AA)
            pl_module.log(f"{loss_name}/{phase}/Kappa_epoch", Kappa)
            # 重置 聚合 metric（注意：重置 class_metrics 即可）
            getattr(pl_module, f"{phase}_{loss_name}_class_metrics").reset()
            # 记录loss
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
            # 保存测试记录
            if pl_module.hparams.config["test_exp_name"] is not None:
                res = "OA: {0:.2f}, AA: {1:.2f}, Kappa: {2:.2f}".format(100 * OA, 100 * AA, 100 * Kappa)
                test_ablation(pl_module, loss_name, res)
            # 累加主metric
            value = OA

        elif loss_name == "mmimdb":
            values = getattr(pl_module, f"{phase}_{loss_name}_F1_scores").compute()
            value = values[1]
            pl_module.log(f"{loss_name}/{phase}/F1_Micro_epoch", values[0])
            pl_module.log(f"{loss_name}/{phase}/F1_Macro_epoch", values[1])
            pl_module.log(f"{loss_name}/{phase}/F1_Samples_epoch", values[2])
            pl_module.log(f"{loss_name}/{phase}/F1_Weighted_epoch", values[3])
            getattr(pl_module, f"{phase}_{loss_name}_F1_scores").reset()

            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()

            if pl_module.hparams.config["test_exp_name"] is not None:
                res = 'F1-Macro: {0:.2f}, F1-Micro: {1:.2f}, F1-Weighted: {2:.2f}, F1-Sample: {3:.2f}'.format(
                    100 * values[1], 100 * values[0], 100 * values[2], 100 * values[3])
                test_ablation(pl_module, loss_name, res)
        elif loss_name == "hatememes":
            value2 = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value2)       
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
            value = getattr(pl_module, f"{phase}_{loss_name}_AUROC").compute()
            pl_module.log(f"{loss_name}/{phase}/AUROC_epoch", value)            
            getattr(pl_module, f"{phase}_{loss_name}_AUROC").reset()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()      
            
            if pl_module.hparams.config["test_exp_name"] is not None:
                res = 'AUROC: {0:.2f}, Accuracy: {1:.2f}'.format(100*value, 100*value2)
                test_ablation(pl_module, loss_name, res)
            
        elif loss_name == "food101":
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value)       
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()

            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()   
            
            if pl_module.hparams.config["test_exp_name"] is not None:
                res = 'Accuracy: {0:.2f}'.format(100*value)
                test_ablation(pl_module, loss_name, res)            
            

            
        elif loss_name == "nlvr2":
            if phase == "train":
                value = getattr(pl_module, f"train_{loss_name}_accuracy").compute()
                pl_module.log(f"{loss_name}/train/accuracy_epoch", value)
                getattr(pl_module, f"train_{loss_name}_accuracy").reset()
                pl_module.log(
                    f"{loss_name}/train/loss_epoch",
                    getattr(pl_module, f"train_{loss_name}_loss").compute(),
                )
                getattr(pl_module, f"train_{loss_name}_loss").reset()
            else:
                value = getattr(pl_module, f"dev_{loss_name}_accuracy").compute()
                pl_module.log(f"{loss_name}/dev/accuracy_epoch", value)
                getattr(pl_module, f"dev_{loss_name}_accuracy").reset()
                pl_module.log(
                    f"{loss_name}/dev/loss_epoch",
                    getattr(pl_module, f"dev_{loss_name}_loss").compute(),
                )
                getattr(pl_module, f"dev_{loss_name}_loss").reset()

                value = getattr(pl_module, f"test_{loss_name}_accuracy").compute()
                pl_module.log(f"{loss_name}/test/accuracy_epoch", value)
                getattr(pl_module, f"test_{loss_name}_accuracy").reset()
                pl_module.log(
                    f"{loss_name}/test/loss_epoch",
                    getattr(pl_module, f"test_{loss_name}_loss").compute(),
                )
                getattr(pl_module, f"test_{loss_name}_loss").reset()
        elif loss_name == "irtr":
            pl_module.log(
                f"{loss_name}/{phase}/irtr_loss_epoch",
                getattr(pl_module, f"{phase}_irtr_loss").compute(),
            )
            getattr(pl_module, f"{phase}_irtr_loss").reset()
        elif loss_name == "mppd" or loss_name == "mpfr":
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
        elif loss_name == "itm":
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
            pl_module.log(
                f"{loss_name}/{phase}/wpa_loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_wpa_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_wpa_loss").reset()
        else:
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()

        the_metric += value

    pl_module.log(f"{phase}/the_metric", the_metric)


def check_non_acc_grad(pl_module):
    if pl_module.token_type_embeddings.weight.grad is None:
        return True
    else:
        grad = pl_module.token_type_embeddings.weight.grad
        return (grad.sum() == 0).item()


def set_task(pl_module):
    pl_module.current_tasks = [
        k for k, v in pl_module.hparams.config["loss_names"].items() if v >= 1
    ]
    return
from transformers import AdamW, get_cosine_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
import torch

def set_schedule(pl_module):
    cfg = pl_module.hparams.config

    print(">>> [Schedule Config] lr =", cfg["learning_rate"], "lr_mult =", cfg["lr_mult"])

    lr = cfg["learning_rate"]
    wd = cfg["weight_decay"]
    lr_mult = cfg["lr_mult"]
    end_lr = cfg["end_lr"]
    decay_power = cfg["decay_power"]
    optim_type = cfg["optim_type"]

    no_decay = [
        "bias",
        "LayerNorm.bias", "LayerNorm.weight",
        "norm.bias", "norm.weight",
        "norm1.bias", "norm1.weight",
        "norm2.bias", "norm2.weight",
    ]
    head_names = [
        "vqa_classifier", "mmimdb_classifier", "food101_classifier",
        "hatememes_classifier", "nlvr2_classifier", "remote_classifier", "embedding_layer"
    ]
    special_names = ["conv_HS", "conv_LIDAR","conv_HS_2", "conv_LIDAR_2"]
    # 过滤掉不需要训练的参数
    def filter_params(cond_fn):
        return [
            p for n, p in pl_module.named_parameters()
            if cond_fn(n, p) and p.requires_grad
        ]

    optimizer_grouped_parameters = [
        {
            "params": filter_params(
                lambda n, p: not any(nd in n for nd in no_decay) and not any(bb in n for bb in head_names) and not any(sn in n for sn in special_names)
            ),
            "weight_decay": wd,
            "lr": lr,
        },
        {
            "params": filter_params(
                lambda n, p: any(nd in n for nd in no_decay) and not any(bb in n for bb in head_names)
            ),
            "weight_decay": 0.0,
            "lr": lr,
        },
        {
            "params": filter_params(
                lambda n, p: not any(nd in n for nd in no_decay) and any(bb in n for bb in head_names) and any(sn in n for sn in special_names)
            ),
            "weight_decay": wd,
            "lr": lr * lr_mult,
        },
        {
            "params": filter_params(
                lambda n, p: any(nd in n for nd in no_decay) and any(bb in n for bb in head_names)
            ),
            "weight_decay": 0.0,
            "lr": lr * lr_mult,
        },
    ]

    # 选择优化器
    if optim_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98)
        )
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
    elif optim_type == "sgd":
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer type: {optim_type}")

    # 计算 max_steps
    if pl_module.trainer.max_steps is None or pl_module.trainer.max_steps == -1:
        max_steps = (
            len(pl_module.trainer.datamodule.train_dataloader())
            * pl_module.trainer.max_epochs
            // pl_module.trainer.accumulate_grad_batches
        )
    else:
        max_steps = pl_module.trainer.max_steps

    # warmup 步数
    warmup_steps = cfg["warmup_steps"]
    if isinstance(warmup_steps, float):
        warmup_steps = int(max_steps * warmup_steps)

    # scheduler
    if decay_power == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
    else:
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
            lr_end=end_lr,
            power=decay_power,
        )

    # ✅ 新 Lightning 推荐返回 dict 格式
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",   # 每 step 更新
            "frequency": 1,
        },
    }

# def set_schedule(pl_module):
#     lr = pl_module.hparams.config["learning_rate"]
#     wd = pl_module.hparams.config["weight_decay"]
#
#     no_decay = [
#         "bias",
#         "LayerNorm.bias",
#         "LayerNorm.weight",
#         "norm.bias",
#         "norm.weight",
#         "norm1.bias",
#         "norm1.weight",
#         "norm2.bias",
#         "norm2.weight",
#     ]
#     head_names = ["remote_classifier", "vqa_classifier", "mmimdb_classifier", "food101_classifier", "hatememes_classifier", "nlvr2_classifier"]
#     prompt_name = "prompt"
#     lr_mult = pl_module.hparams.config["lr_mult"]
#     end_lr = pl_module.hparams.config["end_lr"]
#     decay_power = pl_module.hparams.config["decay_power"]
#     optim_type = pl_module.hparams.config["optim_type"]
#
#     names = [n for n, p in pl_module.named_parameters()]
#
#
#     optimizer_grouped_parameters = [
#         {
#             "params": [
#                 p
#                 for n, p in pl_module.named_parameters()
#                 if not any(nd in n for nd in no_decay)
#                 and not any(bb in n for bb in head_names)
#             ],
#             "weight_decay": wd,
#             "lr": lr,
#         },
#         {
#             "params": [
#                 p
#                 for n, p in pl_module.named_parameters()
#                 if any(nd in n for nd in no_decay)
#                 and not any(bb in n for bb in head_names)
#             ],
#             "weight_decay": 0.0,
#             "lr": lr,
#         },
#         {
#             "params": [
#                 p
#                 for n, p in pl_module.named_parameters()
#                 if not any(nd in n for nd in no_decay)
#                 and any(bb in n for bb in head_names)
#             ],
#             "weight_decay": wd,
#             "lr": lr * lr_mult,
#         },
#         {
#             "params": [
#                 p
#                 for n, p in pl_module.named_parameters()
#                 if any(nd in n for nd in no_decay) and any(bb in n for bb in head_names)
#             ],
#             "weight_decay": 0.0,
#             "lr": lr * lr_mult,
#         },
#     ]
#
#     if optim_type == "adamw":
#         optimizer = AdamW(
#             optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98)
#         )
#     elif optim_type == "adam":
#         optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
#     elif optim_type == "sgd":
#         optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)
#
#     if pl_module.trainer.max_steps is None:
#         max_steps = (
#             len(pl_module.trainer.datamodule.train_dataloader())
#             * pl_module.trainer.max_epochs
#             // pl_module.trainer.accumulate_grad_batches
#         )
#     else:
#         max_steps = pl_module.trainer.max_steps
#
#     warmup_steps = pl_module.hparams.config["warmup_steps"]
#     if isinstance(pl_module.hparams.config["warmup_steps"], float):
#         warmup_steps = int(max_steps * warmup_steps)
#
#     if decay_power == "cosine":
#         scheduler = get_cosine_schedule_with_warmup(
#             optimizer,
#             num_warmup_steps=warmup_steps,
#             num_training_steps=max_steps,
#         )
#     else:
#         scheduler = get_polynomial_decay_schedule_with_warmup(
#             optimizer,
#             num_warmup_steps=warmup_steps,
#             num_training_steps=max_steps,
#             lr_end=end_lr,
#             power=decay_power,
#         )
#
#     sched = {"scheduler": scheduler, "interval": "step"}
#
#     return (
#         [optimizer],
#         [sched],
#     )
