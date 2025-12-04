from .mmimdb_datamodule import MMIMDBDataModule
from .hatememes_datamodule import HateMemesDataModule
from .food101_datamodule import FOOD101DataModule
from .remote_datamodule import REMOTEDataModule
from .remote_datamodule_3 import REMOTEDataModule_3

_datamodules = {
    "mmimdb": MMIMDBDataModule,
    "Hatefull_Memes": HateMemesDataModule,
    "Food101": FOOD101DataModule,
    "remote": REMOTEDataModule,
    "remote_3": REMOTEDataModule_3,
}
