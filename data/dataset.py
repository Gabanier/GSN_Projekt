import torch.nn as nn
from torch.autograd import grad
import lightning as L
from typing import Any, List, Mapping
from utils.utils import sequential_from_descriptor

#TODO
class PendulumDataset(L.LightningDataModule):
    def __init__(self) -> None:
        super().__init__()
