import torch
import random
import os
from .base_dataset import BaseDataset


class LidarHSDataset_3(BaseDataset):
    def __init__(self, *args, split="", missing_info=None, **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["lidarhs_train"]
        elif split == "val":
            names = ["lidarhs_val"]
        else:
            names = ["lidarhs_test"]

        super().__init__(*args, text_column_name="", names=names, **kwargs)

        if missing_info is not None:
            self.simulate_missing = missing_info.get("simulate_missing", False)
            missing_ratio = missing_info["ratio"][split]
            missing_ratio_2 = missing_info["ratio_2"][split]

            missing_type = missing_info["type"][split]
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

                    elif missing_type == "Lidar":
                        missing_table[missing_index] = 2  

                    elif missing_type == "SAR":
                        missing_table[missing_index] = 4  

                    elif missing_type == "HS_Lidar":
                        # LidarHS 顺序 → 先缺 LiDAR，再叠加 HS（2+1=3）
                        missing_table[missing_index] = 2  # missing LiDAR
                        missing_index_2 = random.sample(range(total_num), int(total_num * missing_ratio_2))
                        missing_table[missing_index_2] += 1  # 再叠加 HS → 3

                    elif missing_type == "HS_SAR":
                        missing_table[missing_index] = 1  # missing HS
                        missing_index_2 = random.sample(range(total_num), int(total_num * missing_ratio_2))
                        missing_table[missing_index_2] += 4  # 1+4=5

                    elif missing_type == "SAR_Lidar":
                        missing_table[missing_index] = 4  # missing SAR
                        missing_index_2 = random.sample(range(total_num), int(total_num * missing_ratio_2))
                        missing_table[missing_index_2] += 2  # 4+2=6

                    # === 三模态组合（随机分布六种缺失模式） ===
                    elif missing_type == "HS_SAR_Lidar":
                        for idx in random.sample(range(total_num), int(total_num * missing_ratio)):
                            missing_table[idx] = random.choice([1, 2, 4, 3, 5, 6])

                torch.save(missing_table, missing_table_path)
                print(f"[OK] Missing table generated → {missing_table_path}")

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

    def get_sar(self, index):
        data = self.table["sar"][index].as_py()
        sar = torch.tensor(data, dtype=torch.float32)
        return {"SAR": sar}

    def get_label(self, index):
        label = int(self.table["label"][index].as_py())
        return {"label": label}

    def get_suite(self, index):
        ret = {}
        ret.update(self.get_lidar(index))
        ret.update(self.get_hs(index))
        ret.update(self.get_sar(index))
        ret.update(self.get_label(index))
        if "col" in self.table.column_names:
            ret["col"] = int(self.table["col"][index].as_py())
        if "nul" in self.table.column_names:
            ret["nul"] = int(self.table["nul"][index].as_py())
        return ret

    def __getitem__(self, index):
        sample = self.get_suite(index)
        missing_type = int(self.missing_table[index].item())

        if missing_type in [1, 3, 5]:  # missing HS
            sample["HS"] = torch.ones_like(sample["HS"])
        if missing_type in [2, 3, 6]:  # missing LiDAR
            sample["lidar"] = torch.ones_like(sample["lidar"])
        if missing_type in [4, 5, 6]:  # missing SAR
            sample["SAR"] = torch.ones_like(sample["SAR"])

        sample["missing_type"] = missing_type
        return sample
