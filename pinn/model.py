import torch
import torch.nn as nn
from torch.autograd import grad
import lightning as L
from typing import Any, List, Mapping
from utils.utils import sequential_from_descriptor

class PINN(nn.Module):
    def __init__(self,layer_descriptor:str):
        super().__init__()

        self.seq_layers = sequential_from_descriptor(layer_descriptor) 
    
    def forward(self, x:torch.Tensor):
        pass
    
    def compute_loss(self, x:torch.Tensor):
        pass

    def predict(self, state:torch.Tensor):
        pass


class PINN_PEND(L.LightningModule):
    """
    PyTorch Lightning module for training PINN on Pendulum data.
    """ 
    def __init__(self, layer_descriptor:str, lr:float =1e-3, weight_decay:float = 1e-5) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = PINN(layer_descriptor)
        self.criterion = nn.MSELoss

        self.lr = lr
        self.weight_deay = weight_decay
    
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
    
    
    
    

