import torch
import torch.nn as nn
import numpy as np
import lightning as pl
from typing import Any, Union, Optional, Tuple
from utils.utils import sequential_from_descriptor
from pathlib import Path
from math import pi as PI
from data.dataset import StateRange

class PINC(nn.Module):
    def __init__(self,
                 n_states: int,
                 n_control: int, 
                 horizon_T: float,
                 state_boundaries: StateRange,
                 control_boundaries: StateRange,
                 ) -> None:
        super().__init__()

        self.n_states:int = n_states
        self.n_control:int = n_control
        self.horizon_T:torch.Tensor = torch.tensor(horizon_T,dtype=torch.float16,requires_grad=False).view(-1)
        self.state_boundaries = state_boundaries
        self.control_boundaries = control_boundaries

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
        # Also (B,N,P)
        x, _ = PINC.flatten_points(x)
        t = x[:, 0].clone().requires_grad_(True).view(-1,1)
        y0 = x[:, 1:1 + self.n_states]
        u = x[:, 1 + self.n_states:]

        x_input = torch.cat([t, y0, u], dim=1)
        y = self(x_input)  # (batch_size, n_states)
        y = PINC.denormalize(y, self.state_boundaries)
        u = PINC.denormalize(u, self.control_boundaries)

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
        # Points can also be extended by num samples -> (batch_size, num_samples, point size)
        x, _ = PINC.flatten_points(x)
        targets, _ = PINC.flatten_points(targets)
        y = self(x) 
        mse_y = ((y - targets) ** 2).mean()
        return mse_y

    def predict(self, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        # Predict the next state y[k] = ffn(T,y[k-1], u[k])
        # which can be rewritten as y[l] = ffn(y[k-1],u[k])
        # state: (n_states,) or (1, n_states) current state y[k-1]
        # control: (n_control,) or (1, n_control) control input u[k]
        # Returns: (n_states) predicted next state
        
        state = state.squeeze(0)                 #reshape to [n_states,]
        control = control.squeeze(0)                #reshape to [n_control,]
        
        x = torch.cat([self.horizon_T.to(state.device),state,control])
        with torch.no_grad():
            return self(x).squeeze(0)
        
    @staticmethod
    def flatten_points(x: torch.Tensor) -> Tuple[torch.Tensor,int]:
        #Helper: Flatten (batch_size, n_samples, point_size)
        #  to (B*N, P) for compatibility
        B = int(x.size(0))
        if x.dim() == 3:  # (B, N, P) -> (B*N, P)
            return x.reshape(-1, x.size(-1)), B
        elif x.dim() == 2:  # Already (N,P)
            return x, B
        elif x.dim() == 1:  # 1D: single point (1,P)
            return x.unsqueeze(0), 1
        else:
            raise Exception("Incorrect tensor shape")
        
    @staticmethod
    def normalize(data: torch.Tensor,
                state_boundaries: StateRange
                ) -> torch.Tensor:
        data_shape = data.shape
        if len(data_shape) == 1:
            data = data.unsqueeze(0)        # (1, N)
        elif len(data_shape) > 2:
            raise Exception("Function allows only for arrays with 1 or 2 axis")
        
        nrng: Tuple[float, float] = (-1.0, 1.0)
        n_states = len(state_boundaries)

        for i in range(n_states):
            data_min = state_boundaries[i][0]
            data_max = state_boundaries[i][1]
            data[:, i] = nrng[0] + (nrng[1] - nrng[0]) * (data[:, i] - data_min) / (data_max - data_min + 1e-12)

        return data

    @staticmethod
    def denormalize(data: torch.Tensor,
                    state_boundaries: StateRange,
                    ) -> torch.Tensor:
        data_shape = data.shape
        if len(data_shape) == 1:
            data = data.unsqueeze(0)
        elif len(data_shape) > 2:
            raise Exception("Function allows only for arrays with 1 or 2 axis")
        
        nrng: Tuple[float, float] = (-1.0, 1.0)
        n_states = len(state_boundaries)
        
        for i in range(n_states):
            data_min = state_boundaries[i][0]
            data_max = state_boundaries[i][1]
            data[:, i] = data_min + (data[:,  i] - nrng[0]) * (data_max - data_min) / (nrng[1] - nrng[0] + 1e-12)
        
        return data

class RWP_PINC(PINC):
    def __init__(self, model_descriptor_path:Union[str,Path],
                 n_states: int,
                 n_control: int,
                 horizon_T: float,
                 state_boundaries:StateRange = ((-PI,PI),(-4,4),(-200,200)),
                 control_boundaries: StateRange = ((-0.5, 0.5),),
                 ) -> None:
        super().__init__(n_states, n_control, horizon_T, state_boundaries, control_boundaries)
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

        #scale factors
        self.state_boundaries = state_boundaries
        self.control_boundaries = control_boundaries

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, B = RWP_PINC.flatten_points(x) # handle the 3 distinct cases
        return self.backbone(x).view(B,-1,self.n_states).squeeze(dim=1) # -> (B*N,P) or (B,P) or (1,P)

    def ode_rhs(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        ## Shoud be in the proper state ranges
        ## x1 = (-pi,pi)
        ## x2 = (-4,4)
        ## x3 = (-200,200)
        x,B = RWP_PINC.flatten_points(x)
        u,_ = RWP_PINC.flatten_points(u)

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

        dx = torch.cat([dx_1,dx_2,dx_3],dim=1).view(B,-1,self.n_states).squeeze(dim=1)
        return dx


class RWP_PINC_PL(pl.LightningModule):
    """
    PyTorch Lightning module for training RWP_PINC
    """ 
    def __init__(self,
                 model_descriptor_path:Union[str,Path],
                 n_states: int,
                 n_control: int,
                 horizon_T: float,
                 state_boundaries:StateRange = ((-PI,PI),(-4,4),(-200,200)),
                 control_boundaries: StateRange = ((-0.5, 0.5),),
                 lambda_ic: float = 1,
                 lambda_physics: float = 0.001, 
                 lambda_data: float = 0,
                 scale_factors:Tuple[float,...] = (PI,4.0,200.0),
                 lr: float = 1e-3, 
                 weight_decay: float = 1e-5
                 ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.n_control = n_control
        self.n_states = n_states
        self.horizon_T = horizon_T
        self.lambda_ic = lambda_ic
        self.lambda_physics = lambda_physics
        self.lambda_data = lambda_data
        self.scale_factors = scale_factors
        
        # -- Model -_-
        self.model = RWP_PINC(model_descriptor_path=model_descriptor_path,
                              n_states=n_states,
                              n_control=n_control,
                              horizon_T=horizon_T,
                              state_boundaries=state_boundaries,
                              control_boundaries=control_boundaries)
        
        self.state_boundaries = state_boundaries
        self.control_boundaries = control_boundaries

        # --- Optimizer params ---
        self.lr = lr
        self.weight_decay = weight_decay

        # --- Metrics ---
        # Already included in the base PINC class

    def training_step(self, batches, batch_idx):
        # print(batch)
        # print(len(batch), len(batch[0]))
        loss_physics:Optional[torch.Tensor] = None
        loss_ic:Optional[torch.Tensor] = None
        loss_data:Optional[torch.Tensor] = None
        loss = torch.tensor(0.0, device=self.device)
        for batch in batches:
            # print(batch.keys())
            x = batch['x']
            batch_type = batch['type'][0]

            if batch_type == 'Collocation':
                loss_physics = self.lambda_physics * self.model.compute_physics_loss(x)
                self.log('train_loss_physics', loss_physics)
            elif batch_type == 'IC':
                targets = batch['targets']
                loss_ic = self.lambda_ic * self.model.compute_data_loss(x=x, targets=targets)
                self.log('train_loss_ic', loss_ic)
            elif batch_type == "ExperimentData":
                targets = batch['targets']
                loss_data = self.lambda_data * self.model.compute_data_loss(x=x, targets=targets)
                self.log('train_loss_exp', loss_data)
            else:
                raise Exception(f"Train-Step: Unknown batch type\nCorrect = [Collocation, IC, ExperimentData]\nGot {batch_type}")

        # Global loss accumulation
        if loss_data is None:
            raise Exception()
        if loss_ic is None:
            raise Exception()
        if loss_physics is None:
            raise Exception()
        
        loss += loss_data + loss_ic + loss_physics
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        ## TODO actual validation implementation
        ## of the discrete time step inference
        x = batch['x']
        batch_type = batch['type'][0]
        targets = batch['targets']
        loss = torch.tensor(0.0, device=self.device)
        if batch_type == "ExperimentData":
            loss = self.model.compute_data_loss(x=x, targets=targets)
        else:
            raise Exception(f"Valid-Step: Unknown batch type\nCorrect = [ExperimentData]\nGot {batch_type}")

        self.log('valid_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        ## TODO actual test implementation
        ## of the discrete time step inference
        x = batch['x']
        batch_type = batch['type'][0]
        targets = batch['targets']
        loss = torch.tensor(0.0, device=self.device)
        if batch_type == "ExperimentData":
            loss = self.model.compute_data_loss(x=x, targets=targets)
        else:
            raise Exception(f"Test-Step: Unknown batch type\nCorrect = [ExperimentData]\nGot {batch_type}")

        self.log('test_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
