import torch
import torch.nn as nn
import lightning as L
from typing import Any, Union, Optional, Tuple
from utils.utils import sequential_from_descriptor
from pathlib import Path

class PINC(nn.Module):
    def __init__(self,
                 n_states: int,
                 n_control: int, 
                 horizon_T: float,
                 lambda_ic:float=1,
                 lambda_physics:float=1e-3,
                 lambda_data:float=0.) -> None:
        super().__init__()

        self.n_states:int = n_states
        self.n_control:int = n_control
        self.horizon_T:torch.Tensor = torch.tensor(horizon_T,dtype=torch.float16,requires_grad=False).view(-1)
        self.lambda_ic:float = lambda_ic
        self.lambda_physics:float = lambda_physics
        self.lambda_data:float = lambda_data
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, (t, y(0){1,..,k},u{1,..,j}) ) where
        # t            - time in range [0,T]
        # y(0){1,..,k} - initial conditions of the system
        # u{1,..,j}    - constant control signal 
        raise NotImplementedError("Each subclass of PINC should implement their own forward pass")
    
    
    def ode_rhs(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # Subclass and implement the right-hand side of the ODE: dx/dt = f(x, u)
        # x: (batch_size, n_states) - states of the system
        # u: (batch_size, n_control) - control inputs
        raise NotImplementedError("Each subclass of PINC should implement their own rhs odes")
    
    def compute_physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        # Computes the physics-informed loss for collocation points.
        # x: (batch_size, (t, y(0){1,..,k}, u{1,..,j}) )
        if x.ndim == 1:
            t = x[0].clone().view(1,-1).requires_grad_(True) # (1, 1)
            y0 = x[1:1 + self.n_states].view(1,-1)  # (1, n_states)
            u = x[1 + self.n_states:].view(1,-1)  # (1, n_control)
        else:
            t = x[:, 0].clone().requires_grad_(True).view(-1,1)  # (batch_size, 1)
            y0 = x[:, 1:1 + self.n_states]  # (batch_size, n_states)
            u = x[:, 1 + self.n_states:]  # (batch_size, n_control)

        x_input = torch.cat([t, y0, u], dim=1)
        y = self(x_input)  # (batch_size, n_states)
        
        # Compute dy/dt
        dy_dt = torch.zeros_like(y)
        for i in range(self.n_states):
            grad_outputs = torch.zeros_like(y)
            grad_outputs[:, i] = 1.0
            dy_i_dt = torch.autograd.grad(
                y[:, i], t, grad_outputs=grad_outputs[:, i:i+1].view(-1),
                create_graph=True
            )[0]
            dy_dt[:, i:i+1] = dy_i_dt

        # Compute the RHS 
        f = self.ode_rhs(y, u)
        mse_f = ((dy_dt - f) ** 2).mean()
        return mse_f

    def compute_data_loss(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Can compute both IC and target data loss.
        # x: (batch_size, (t, y(0){1,..,k}, u{1,..,j}) )
        # Labels are either
        # targets: (batch_size, y(0){1,..,k}) which is the appropriate structure
        # to compute the IC loss. 
        # OR
        # targets: (batch_size, y(t){1,...,k}) which is the appropriate structure
        # to compute the DATA loss.
        y = self(x) 
        mse_y = ((y - targets) ** 2).mean()
        return mse_y
    
    def compute_total_loss(self, x_coll: torch.Tensor, 
                           x_train: torch.Tensor, 
                           targets_ic: torch.Tensor,
                           x_data: Optional[torch.Tensor] = None,
                           targets_data: Optional[torch.Tensor] = None) ->Tuple[torch.Tensor,Tuple[torch.Tensor,torch.Tensor,Optional[torch.Tensor]]]:
       
        loss_physics = self.lambda_physics*self.compute_physics_loss(x_coll)
        loss_ic = self.lambda_ic*self.compute_data_loss(x=x_train, targets=targets_ic)
        if x_data is not None and targets_data is not None:
            loss_data = self.lambda_data*self.compute_data_loss(x_data, targets_data)
        else:
            loss_data = torch.zeros_like(loss_physics)
        return (loss_physics + loss_ic + loss_data, (loss_physics, loss_ic, loss_data))

    def predict(self, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        # Predict the next state y[k] = ffn(T,y[k-1], u[k])
        # which can be rewritten as y[l] = ffn(y[k-1],u[k])
        # state: (n_states,) or (1, n_states) current state y[k-1]
        # control: (n_control,) or (1, n_control) control input u[k]
        # Returns: (n_states) predicted next state
        
        state = state.squeeze(0)                 #reshape to [n_states,]
        control = control.squeeze(0)                #reshape to [n_control,]
        # t       = self.horizon_T.to(state.device) #send T to device [1,1]
        
        x = torch.cat([self.horizon_T.to(state.device),state,control])
        with torch.no_grad():
            return self(x).squeeze(0)


class RWP_PINC(PINC):
    def __init__(self, model_descriptor_path:Union[str,Path],
                 n_states: int,
                 n_control: int,
                 horizon_T: float,
                 lambda_ic: float = 1,
                 lambda_physics: float = 0.001, 
                 lambda_data: float = 0) -> None:
        super().__init__(n_states, n_control, horizon_T, lambda_ic, lambda_physics, lambda_data)
        self.backbone = sequential_from_descriptor(model_descriptor_path)
        self.k_t   = 0.027        #% stała momentu silnika [Nm/A]
        self.k_emf = self.k_t     #% stała SEM [Vs/rad]
        self.R     = 2.3          #% rezystancja [Ω]
        self.k_p   = 10.5         #% nachylenie tanh dla wahadła
        voltage_gain = 12
        
        self.b_1   = 8.7462e-04   #% tarcie lepkie – wahadło [Nm/(rad/s)]
        self.b_2   = 0.0032       #% tarcie Coulomba – wahadło [Nm]
        self.b_3   = (self.k_t / self.R) * voltage_gain      #% wzmocnienie prąd → moment
        self.b_4   = (self.k_emf * self.k_t) / self.R  #% sprzężenie zwrotne
        self.b_5 = 0.7991;         #% sprawność silnika
        
        # % Dodatkowe parametry [nieużywane]
        bm_sf = 0.0027             #% tarcie Coulomba – silnik [Nm]
        bm_df = 0                  #% tarcie lepkie Silnika [Nm/(rad/s)]
        k_m = 10                   #% nachylenie tanh dla Silnika

        # % Parametry mechaniczne
        self.J_r = 2.8713e-04     #% moment bezwładności wirnika [kg·m²]
        self.J_p = 0.0274         #% moment bezwładności wahadła [kg·m²]
        md  = 0.01218               #% masa wahadła * odległość [kg*m]
        self.G   = md * 9.81      #% siła ciężkości [N]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def ode_rhs(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        x_1 = x[:,0].view(-1,1) #phi - Pendulum Angle
        x_2 = x[:,1].view(-1,1) #dphi - Pendulum Velocity
        x_3 = x[:,2].view(-1,1) #dtheta - DC Velocity
      
        #% Moment silnika
        tau_m = self.b_3 * u - self.b_4 * x_3

        #% Równania dynamiki
        dx_1 = x_2
        dx_2 = (-self.G * torch.sin(x_1) 
                - self.b_1 * x_2 
                - self.b_2 * torch.tanh(self.k_p * x_2) 
                - self.b_5 * tau_m) / self.J_p
        dx_3 = tau_m / self.J_r
        return torch.cat([dx_1,dx_2,dx_3],dim=1)


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
