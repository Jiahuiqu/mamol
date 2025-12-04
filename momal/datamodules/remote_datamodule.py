from LMMOE.datasets import LidarHSDataset
from .datamodule_base import BaseDataModule
from collections import defaultdict


class REMOTEDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return LidarHSDataset

    @property
    def dataset_name(self):
        return "remote"

    def setup(self, stage):
        super().setup(stage)

