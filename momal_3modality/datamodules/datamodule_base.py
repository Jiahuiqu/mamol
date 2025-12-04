import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class BaseDataModule(LightningDataModule):
    def __init__(self, _config):
        super().__init__()

        self.data_dir = _config["data_root"]

        self.num_workers = _config["num_workers"]
        self.batch_size = _config["per_gpu_batchsize"]
        self.eval_batch_size = self.batch_size

        self.image_size = _config["image_size"]
        self.max_text_len = _config["max_text_len"]
        self.draw_false_image = _config["draw_false_image"]
        self.draw_false_text = _config["draw_false_text"]
        self.image_only = _config["image_only"]

        # ====== 缺失模态信息 ======
        self.missing_info = {
            'ratio': _config["missing_ratio"],
            'ratio_2': _config.get("missing_ratio_2", None),  # ✅ 新增这一行
            'type': _config["missing_type"],
            'both_ratio': _config["both_ratio"],
            'missing_table_root': _config["missing_table_root"],
            'simulate_missing': _config["simulate_missing"]
        }

        # for bash execution (测试时覆盖 val/test)
        if _config.get("test_ratio") is not None:
            self.missing_info['ratio']['val'] = _config["test_ratio"]
            self.missing_info['ratio']['test'] = _config["test_ratio"]
        if _config.get("test_ratio_2") is not None:
            self.missing_info['ratio_2']['val'] = _config["test_ratio_2"]
            self.missing_info['ratio_2']['test'] = _config["test_ratio_2"]
        if _config.get("test_type") is not None:
            self.missing_info['type']['val'] = _config["test_type"]
            self.missing_info['type']['test'] = _config["test_type"]

        # transforms
        self.train_transform_keys = (
            ["default_train"]
            if len(_config["train_transform_keys"]) == 0
            else _config["train_transform_keys"]
        )
        self.val_transform_keys = (
            ["default_val"]
            if len(_config["val_transform_keys"]) == 0
            else _config["val_transform_keys"]
        )

        self.setup_flag = False

    # ========= 子类接口 =========
    @property
    def dataset_cls(self):
        raise NotImplementedError("return dataset class")

    @property
    def dataset_name(self):
        raise NotImplementedError("return name of dataset")

    # ========= dataset 构建 =========
    def set_train_dataset(self):
        self.train_dataset = self.dataset_cls(
            self.data_dir,
            self.train_transform_keys,
            split="train",
            image_size=self.image_size,
            draw_false_image=self.draw_false_image,
            draw_false_text=self.draw_false_text,
            image_only=self.image_only,
            missing_info=self.missing_info,
        )

    def set_val_dataset(self):
        self.val_dataset = self.dataset_cls(
            self.data_dir,
            self.val_transform_keys,
            split="val",
            image_size=self.image_size,
            draw_false_image=self.draw_false_image,
            draw_false_text=self.draw_false_text,
            image_only=self.image_only,
            missing_info=self.missing_info,
        )

    def set_test_dataset(self):
        self.test_dataset = self.dataset_cls(
            self.data_dir,
            self.val_transform_keys,
            split="test",
            image_size=self.image_size,
            draw_false_image=self.draw_false_image,
            draw_false_text=self.draw_false_text,
            image_only=self.image_only,
            missing_info=self.missing_info,
        )

    # ========= setup =========
    def setup(self, stage=None):
        if not self.setup_flag:
            self.set_train_dataset()
            self.set_val_dataset()
            self.set_test_dataset()
            self.setup_flag = True

    # ========= dataloader =========
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.train_dataset.collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.val_dataset.collate,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.test_dataset.collate,
        )
