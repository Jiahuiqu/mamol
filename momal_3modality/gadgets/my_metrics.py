import torch
from torchmetrics.functional import f1_score, auroc
# from pytorch_lightning.metrics import Metric
from torchmetrics import Metric

# class ClassificationMetrics(Metric):
#     """
#     计算 OA（Overall Accuracy）, AA（Average Accuracy）, Kappa（Cohen’s Kappa）
#     """
#     def __init__(self, num_classes, dist_sync_on_step=False):
#         super().__init__(dist_sync_on_step=dist_sync_on_step)
#         self.num_classes = num_classes
#         self.add_state("confmat", default=torch.zeros(num_classes, num_classes, dtype=torch.long), dist_reduce_fx="sum")
#
#     def update(self, preds, target):
#         preds, target = preds.detach().to(self.confmat.device), target.detach().to(self.confmat.device)
#         if preds.ndim > 1:
#             preds = preds.argmax(dim=1)
#
#         mask = (target >= 0) & (target < self.num_classes)
#         preds, target = preds[mask], target[mask]
#
#         cm = torch.zeros_like(self.confmat)
#         indices = torch.stack([target, preds])
#         cm.index_put_(tuple(indices), torch.tensor(1, device=self.confmat.device), accumulate=True)
#         self.confmat += cm
#
#     def compute(self):
#         confmat = self.confmat.float()
#         diag = torch.diag(confmat)
#         total = confmat.sum()
#         per_class_total = confmat.sum(dim=1)
#
#         # OA
#         OA = diag.sum() / (total + 1e-8)
#         # AA
#         AA = torch.mean(diag / (per_class_total + 1e-8))
#         # Kappa
#         pe = torch.sum(confmat.sum(0) * confmat.sum(1)) / (total ** 2 + 1e-8)
#         Kappa = (OA - pe) / (1 - pe + 1e-8)
#         return {"OA": OA, "AA": AA, "Kappa": Kappa}
class ClassificationMetrics(Metric):
    """
    计算 OA（Overall Accuracy）, AA（Average Accuracy）, Kappa（Cohen’s Kappa）
    使用 bincount 来构建混淆矩阵，避免 index_put 的 shape 问题。
    """
    def __init__(self, num_classes, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_classes = int(num_classes)
        self.add_state(
            "confmat",
            default=torch.zeros(self.num_classes, self.num_classes, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    def update(self, preds, target):
        # move to device and ensure long dtype
        device = self.confmat.device
        preds = preds.detach().to(device)
        target = target.detach().to(device)

        # If logits/probabilities passed, convert to preds
        if preds.ndim > 1 and preds.size(-1) > 1:
            preds = preds.argmax(dim=1)
        elif preds.ndim > 1 and preds.size(-1) == 1:
            # shape [B,1] -> [B]
            preds = preds.view(-1)

        # Flatten to 1D
        preds = preds.view(-1).long()
        target = target.view(-1).long()

        # Mask invalid labels (e.g. -100) and ensure in range
        mask = (target >= 0) & (target < self.num_classes)
        if mask.sum() == 0:
            return  # nothing to update
        preds = preds[mask]
        target = target[mask]

        # Extra safety: ensure preds are also in valid range
        preds = preds % self.num_classes  # or optionally mask out-of-range

        # compute linear indices and bincount
        idx = (target * self.num_classes + preds).to(torch.long)
        # bincount = torch.bincount(idx, minlength=self.num_classes * self.num_classes).to(device)
        bincount = torch.bincount(idx.cpu(), minlength=self.num_classes * self.num_classes).to(device)

        # reshape and add
        cm = bincount.view(self.num_classes, self.num_classes).to(torch.long)
        self.confmat += cm

    def compute(self):
        confmat = self.confmat.float()
        diag = torch.diag(confmat)
        total = confmat.sum()
        per_class_total = confmat.sum(dim=1)

        OA = diag.sum() / (total + 1e-8)
        AA = torch.mean(diag / (per_class_total + 1e-8))
        pe = torch.sum(confmat.sum(0) * confmat.sum(1)) / (total ** 2 + 1e-8)
        Kappa = (OA - pe) / (1 - pe + 1e-8)
        return {"OA": OA, "AA": AA, "Kappa": Kappa}

class Accuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):
        logits, target = (
            logits.detach().to(self.correct.device),
            target.detach().to(self.correct.device),
        )
        if logits.size(-1)>1:
            preds = logits.argmax(dim=-1)
        else:
            preds = (torch.sigmoid(logits)>0.5).long()
            
        preds = preds[target != -100]
        target = target[target != -100]
        if target.numel() == 0:
            return 1

        assert preds.shape == target.shape

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct / self.total

class AUROC(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("logits", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, logits, target):
        logits, targets = (
            logits.detach().to(self.correct.device),
            target.detach().to(self.correct.device),
        )
        
        self.logits.append(logits)
        self.targets.append(targets)


    def compute(self):
        if type(self.logits) == list:
            all_logits = torch.cat(self.logits)
            all_targets = torch.cat(self.targets).long()
        else:
            all_logits = self.logits
            all_targets = self.targets.long()
    
        if all_logits.size(-1)>1:
            all_logits = torch.softmax(all_logits, dim=1)
            AUROC = auroc(all_logits, all_targets, num_classes=2)
        else:
            all_logits = torch.sigmoid(all_logits)
            AUROC = auroc(all_logits, all_targets)
        
        return AUROC
    
class F1_Score(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("logits", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, logits, target):
        logits, targets = (
            logits.detach().to(self.correct.device),
            target.detach().to(self.correct.device),
        )
        
        self.logits.append(logits)
        self.targets.append(targets)


    def compute(self, use_sigmoid=True):
        if type(self.logits) == list:
            all_logits = torch.cat(self.logits)
            all_targets = torch.cat(self.targets).long()
        else:
            all_logits = self.logits
            all_targets = self.targets.long()
        if use_sigmoid:
            all_logits = torch.sigmoid(all_logits)
        F1_Micro = f1_score(all_logits, all_targets, average='micro')
        F1_Macro = f1_score(all_logits, all_targets, average='macro', num_classes=23)
        F1_Samples = f1_score(all_logits, all_targets, average='samples')
        F1_Weighted = f1_score(all_logits, all_targets, average='weighted', num_classes=23)
        return (F1_Micro, F1_Macro, F1_Samples, F1_Weighted)
    
class check(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("logits", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, logits, target):
        logits, targets = (
            logits.detach().to(self.correct.device),
            target.detach().to(self.correct.device),
        )
        
        self.logits.append(logits)
        self.targets.append(targets)


    def compute(self, use_sigmoid=True):
        if type(self.logits) == list:
            all_logits = torch.cat(self.logits).long()
            all_targets = torch.cat(self.targets).long()
        else:
            all_logits = self.logits.long()
            all_targets = self.targets.long()

        mislead = all_logits ^ all_targets
        accuracy = mislead.sum(dim=0)
        return accuracy
        
class Scalar(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("scalar", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, scalar):
        if isinstance(scalar, torch.Tensor):
            scalar = scalar.detach().to(self.scalar.device)
        else:
            scalar = torch.tensor(scalar).float().to(self.scalar.device)
        self.scalar += scalar
        self.total += 1

    def compute(self):
        return self.scalar / self.total    
    
class Scalar2(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("scalar", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, scalar, num):
        if isinstance(scalar, torch.Tensor):
            scalar = scalar.detach().to(self.scalar.device)
        else:
            scalar = torch.tensor(scalar).float().to(self.scalar.device)
        
        self.scalar += scalar
        self.total += num

    def compute(self):
        return self.scalar / self.total


class VQAScore(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits, target):
        logits, target = (
            logits.detach().float().to(self.score.device),
            target.detach().float().to(self.score.device),
        )
        logits = torch.max(logits, 1)[1]
        one_hots = torch.zeros(*target.size()).to(target)
        one_hots.scatter_(1, logits.view(-1, 1), 1)
        scores = one_hots * target

        self.score += scores.sum()
        self.total += len(logits)

    def compute(self):
        return self.score / self.total
