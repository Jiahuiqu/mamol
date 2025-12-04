import random
import torch
import io
import pyarrow as pa
import os
from PIL import Image
from LMMOE.transforms import keys_to_transforms  
import numpy as np


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        transform_keys: list,
        image_size: int = 224,
        names: list = [],
        text_column_name: str = "",
        remove_duplicate=True,
        max_text_len=40,
        draw_false_image=0,
        draw_false_text=0,
        image_only=False,
        missing_ratio={},
        missing_type={},
    ):

        assert len(transform_keys) >= 1
        super().__init__()

        self.transforms = keys_to_transforms(transform_keys, size=image_size)
        self.text_column_name = text_column_name
        self.names = names
        self.max_text_len = max_text_len
        self.draw_false_image = draw_false_image
        self.draw_false_text = draw_false_text
        self.image_only = image_only
        self.data_dir = data_dir

        if len(names) != 0:
            tables = [
                pa.ipc.RecordBatchFileReader(
                    pa.memory_map(f"{data_dir}/{name}.arrow", "r")
                ).read_all()
                for name in names
                if os.path.isfile(f"{data_dir}/{name}.arrow")
            ]

            self.table_names = sum([[name] * len(tables[i]) for i, name in enumerate(names)], [])
            self.table = pa.concat_tables(tables, promote=True)

            if text_column_name != "":
                self.all_texts = self.table[text_column_name].to_pandas().tolist()
                if remove_duplicate:
                    self.all_texts = [list(set(texts)) for texts in self.all_texts]
            else:
                self.all_texts = []
        else:
            self.table = None
            self.all_texts = []

        self.index_mapper = {}
        if self.table is not None:
            if text_column_name != "" and not self.image_only:
                j = 0
                for i, texts in enumerate(self.all_texts):
                    for _j in range(len(texts)):
                        self.index_mapper[j] = (i, _j)
                        j += 1
            else:
                for i in range(len(self.table)):
                    self.index_mapper[i] = (i, None)

    @property
    def corpus(self):
        return [text for texts in self.all_texts for text in texts]

    def __len__(self):
        return len(self.index_mapper) if self.table is not None else 0

    def get_raw_image(self, index, image_key="image"):
        index, _ = self.index_mapper[index]
        image_bytes = io.BytesIO(self.table[image_key][index].as_py())
        image_bytes.seek(0)
        return Image.open(image_bytes).convert("RGB")

    def get_image(self, index, image_key="image"):
        image = self.get_raw_image(index, image_key=image_key)
        image_tensor = [tr(image) for tr in self.transforms]  
        return {
            "image": image_tensor,
            "img_index": self.index_mapper[index][0],
            "cap_index": self.index_mapper[index][1],
            "raw_index": index,
        }

    def get_lidarHS(self, lidar, hs):
        # 支持自定义 transform key
        if "lidarHS_transform" in self.transforms:
            return self.transforms["lidarHS_transform"]({"lidar": lidar, "HS": hs})
        else:
            return {"lidar": lidar, "HS": hs}

    def get_text(self, raw_index):
        index, caption_index = self.index_mapper[raw_index]
        text = self.all_texts[index][caption_index]
        return {
            "text": text,
            "img_index": index,
            "cap_index": caption_index,
            "raw_index": raw_index,
        }

    def get_suite(self, index):
        result = None
        while result is None:
            try:
                ret = {}
                if self.table is not None:  
                    ret.update(self.get_image(index))
                    if not self.image_only and len(self.all_texts) > 0:
                        txt = self.get_text(index)
                        ret.update({"replica": True if txt["cap_index"] > 0 else False})
                        ret.update(txt)
                result = True
            except Exception as e:
                print(f"Error reading idx {index}: {e}")
                index = random.randint(0, len(self.index_mapper) - 1)
        return ret

    def collate(self, batch):
        batch_size = len(batch)
        keys = set(k for b in batch for k in b.keys())
        dict_batch = {k: [dic.get(k) for dic in batch] for k in keys}

        rgb_keys = [k for k in dict_batch if "rgb" in k or "image" in k]
        hs_keys = [k for k in dict_batch if "HS" in k or "hs" in k]
        lidar_keys = [k for k in dict_batch if "lidar" in k or "LiDAR" in k]

        def pad_tensors(tensor_list, dim=3):
            shapes = [t.shape for t in tensor_list if t is not None]
            max_shape = [max(s[i] for s in shapes) for i in range(dim)]
            padded = torch.zeros(len(tensor_list), *max_shape)
            for i, t in enumerate(tensor_list):
                if t is None:
                    continue
                slices = tuple(slice(0, s) for s in t.shape)
                padded[i][slices] = t
            return padded

        for k in rgb_keys:
            imgs = [x[0] if isinstance(x, list) else x for x in dict_batch[k]]
            dict_batch[k] = pad_tensors(imgs, dim=3)

        for k in hs_keys:
            hs_imgs = dict_batch[k]
            dict_batch[k] = pad_tensors(hs_imgs, dim=3)

        for k in lidar_keys:
            lidar_maps = dict_batch[k]
            dict_batch[k] = pad_tensors(lidar_maps, dim=len(lidar_maps[0].shape))

        if "label" in dict_batch:
            dict_batch["label"] = torch.tensor(dict_batch["label"], dtype=torch.long)

        return dict_batch
