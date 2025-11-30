import torch
import torch.nn as nn
from torch.autograd import grad
import lightning as L
from typing import Any, List, Mapping
from utils.utils import sequential_from_descriptor

#TODO
class PINC(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x:torch.Tensor):
        pass
    
    def compute_loss(self, x:torch.Tensor):
        pass

    def predict(self, state:torch.Tensor):
        pass


#TODO
class LightningPINC(L.LightningModule):
    """
    PyTorch Lightning module for training PINN on Pendulum data.
    """ 
    def __init__(self) -> None:
        super().__init__()
        self.save_hyperparameters()
    
    def step(self, batch):
        pass

    def training_step(self, *args: Any, **kwargs: Any):
        return super().training_step(*args, **kwargs)
    
    def validation_step(self, *args: Any, **kwargs: Any):
        return super().validation_step(*args, **kwargs)
    
    def test_step(self, *args: Any, **kwargs: Any):
        return super().test_step(*args, **kwargs)
    
    def configure_optimizers(self):
        return super().configure_optimizers()
    