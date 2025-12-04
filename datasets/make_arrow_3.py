import os
import json
import pandas as pd
import pyarrow as pa
import numpy as np
from tqdm import tqdm
from dataloader_agusburg import LIDARHS


def make_arrow_lidarhs(dataset, dataset_root, split="train"):
    data_list = []
    for idx in tqdm(range(len(dataset))):
        sample = dataset[idx]

        # 统一接口
        lidar = sample["lidar"].numpy().tolist()
        HS = sample["HS"].numpy().tolist()
        sar = sample["sar"].numpy().tolist()
        label = int(sample["label"])
        col, nul = int(sample["ik"][0]), int(sample["ik"][1])

        data = (lidar, HS, sar, label, col, nul)
        data_list.append(data)

    dataframe = pd.DataFrame(
        data_list,
        columns=["lidar", "HS", "sar", "label", "col", "nul"],
    )

    table = pa.Table.from_pandas(dataframe, preserve_index=False)
    os.makedirs(dataset_root, exist_ok=True)
    out_path = f"{dataset_root}/lidarhs_{split}.arrow"
    with pa.OSFile(out_path, "wb") as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)

    print(f"[OK] Saved {split} split to {out_path}, total {len(dataframe)} samples.")

if __name__ == "__main__":
    save_root = "./lidarhs_arrow"

    # train
    train_dataset = LIDARHS(patchsize=14, mode="train", classnum=100)
    make_arrow_lidarhs(train_dataset, save_root, split="train")

    # test
    test_dataset = LIDARHS(patchsize=14, mode="train", classnum=300)
    make_arrow_lidarhs(test_dataset, save_root, split="test")
