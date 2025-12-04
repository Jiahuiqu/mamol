
import torch
import random
import os
from .base_dataset import BaseDataset

class LidarHSDataset(BaseDataset):
    def __init__(self, *args, split="", missing_info=None, **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["lidarhs_train"]
        elif split == "val":
            names = ["lidarhs_val"]
        else:
            names = ["lidarhs_test"]

        super().__init__(*args, text_column_name="",  names=names, **kwargs)

        if missing_info is not None:
            self.simulate_missing = missing_info.get("simulate_missing", False)
            missing_ratio = missing_info["ratio"][split]
            missing_type = missing_info["type"][split]
            both_ratio = missing_info.get("both_ratio", 0.5)
            missing_table_root = missing_info["missing_table_root"]
            os.makedirs(missing_table_root, exist_ok=True)

            mratio = str(missing_ratio).replace('.', '')
            missing_table_name = f'{names[0]}_missing_{missing_type}_{mratio}.pt'
            missing_table_path = os.path.join(missing_table_root, missing_table_name)

            total_num = len(self.table["lidar"])

            if os.path.exists(missing_table_path):
                missing_table = torch.load(missing_table_path)
                if len(missing_table) != total_num:
                    print("Warning: missing table size mismatch, regenerating...")
                    os.remove(missing_table_path)
                    missing_table = None
            else:
                missing_table = None

            if missing_table is None:
                missing_table = torch.zeros(total_num)
                if missing_ratio > 0:
                    missing_index = random.sample(range(total_num), int(total_num * missing_ratio))
                    if missing_type == "HS":
                        missing_table[missing_index] = 1
                    elif missing_type == "lidar":
                        missing_table[missing_index] = 2
                    elif missing_type == "both":
                        missing_table[missing_index] = 1
                        missing_index_lidar = random.sample(missing_index, int(len(missing_index) * both_ratio))
                        missing_table[missing_index_lidar] = 2
                torch.save(missing_table, missing_table_path)

            self.missing_table = missing_table
        else:
            self.simulate_missing = False
            self.missing_table = torch.zeros(len(self.table["lidar"]))

    def get_lidar(self, index):
        data = self.table["lidar"][index].as_py()
        lidar = torch.tensor(data, dtype=torch.float32)
        return {"lidar": lidar}

    def get_hs(self, index):
        data = self.table["HS"][index].as_py()
        hs = torch.tensor(data, dtype=torch.float32)
        return {"HS": hs}

    def get_label(self, index):
        label = int(self.table["label"][index].as_py())
        return {"label": label}

    def get_suite(self, index):
        ret = {}
        ret.update(self.get_lidar(index))
        ret.update(self.get_hs(index))
        ret.update(self.get_label(index))
        if "col" in self.table.column_names:
            ret["col"] = int(self.table["col"][index].as_py())
        if "nul" in self.table.column_names:
            ret["nul"] = int(self.table["nul"][index].as_py())
        return ret

    def __getitem__(self, index):
        sample = self.get_suite(index)

        missing_type = self.missing_table[index].item()

        simulate_missing_type = 0
        if self.split == "train" and self.simulate_missing and missing_type == 0:
            simulate_missing_type = random.choice([0, 1, 2])

        if missing_type == 1 or simulate_missing_type == 1:
            sample["HS"] = torch.ones_like(sample["HS"])

        if missing_type == 2 or simulate_missing_type == 2:
            sample["lidar"] = torch.ones_like(sample["lidar"])

        sample["missing_type"] = missing_type + simulate_missing_type

        return sample
