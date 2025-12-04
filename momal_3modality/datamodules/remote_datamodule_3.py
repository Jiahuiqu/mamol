from LMMOE_3modality.datasets import LidarHSDataset_3
from .datamodule_base import BaseDataModule
from collections import defaultdict


class REMOTEDataModule_3(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return LidarHSDataset_3

    @property
    def dataset_name(self):
        return "remote_3"

    def setup(self, stage):
        super().setup(stage)

