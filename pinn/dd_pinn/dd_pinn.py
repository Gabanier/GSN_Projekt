import torch
import torch.nn as nn
from torch.autograd import grad
import lightning as L
from typing import Any, List, Mapping


import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from typing import Any, Optional, Tuple


class DDPINN(nn.Module):
    """Domain-Decoupled PINN implementation for a 3-state pendulum-like system. 
    The input tensor should have shape (B, D), where D = 1 (timestamp) + 1 (u) + 3 (state) = 5, 
    with each row in the format [t, u, theta, omega, phi]. The forward method returns x_hat(t) = x0 + g(a, t) 
    with shape (B, m) where m=3. The compute_loss method returns the physics loss computed as 
    MSE(xdot_hat, f_ssm(x_hat, u(t))). The f_ssm method should be replaced with the precise dynamics 
    of your pendulum+flywheel system."""

    def __init__(
        self,
        m: int = 3,
        n_g: int = 16,
        hidden_dims: Tuple[int, ...] = (128, 128),
        activation: nn.Module = nn.GELU()
    ):
        super().__init__()
        self.m = m
        self.n_g = n_g
        self.phi = torch.sin
        self.dphi = torch.cos
        self.loss_phy_weight = 0.5
        self.n_params_per_subfn = 3
        self.out_dim = self.n_params_per_subfn * self.m * self.n_g
        in_dim = 1 + 1 + self.m
        mlp_in_dim = 1 + self.m
        layers = []
        prev = mlp_in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(activation)
            prev = h
        layers.append(nn.Linear(prev, self.out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Takes as input a batch tensor of shape (B, 5) in the format [t, u, theta, omega, phi] 
        and returns x_hat(t) of shape (B, m)."""
        assert batch.ndim == 2 and batch.shape[1] >= (1 + 1 + self.m)
        t = batch[:, 0].unsqueeze(-1)
        u = batch[:, 1].unsqueeze(-1)
        x0 = batch[:, 2:2+self.m]
        f_in = torch.cat([u, x0], dim=-1)
        a_raw = self.mlp(f_in)
        B = a_raw.shape[0]
        a = a_raw.view(B, self.m, self.n_g, self.n_params_per_subfn)
        alpha = a[..., 0]
        beta  = a[..., 1]
        gamma = a[..., 2]
        beta = F.softplus(beta) + 1e-6
        alpha = torch.tanh(alpha)
        alpha = alpha * 2.0
        t_expand = t.view(B, 1, 1).expand(-1, self.m, self.n_g)
        arg = beta * t_expand + gamma
        phi_arg = torch.sin(arg)
        phi_gamma = torch.sin(gamma)
        g = (alpha * (phi_arg - phi_gamma)).sum(dim=-1)
        x_hat = x0 + g
        return x_hat

    def derivative_closed_form(self, batch: torch.Tensor) -> torch.Tensor:
        """Computes the analytical time derivative ˙g(a,t) for the current batch, i.e. 
        ˙g_j(t) = sum_i alpha_ij * beta_ij * φ'(beta_ij t + gamma_ij). For φ = sin, φ' = cos. 
        Returns xdot_hat (B, m) which equals ˙g since x0 is constant. 
        The same transforms as in forward() are applied."""
        assert batch.ndim == 2 and batch.shape[1] >= (1 + 1 + self.m)
        t = batch[:, 0].unsqueeze(-1)           # (B,1)
        u = batch[:, 1].unsqueeze(-1)           # (B,1)
        x0 = batch[:, 2:2+self.m]               # (B,m)

        f_in = torch.cat([u, x0], dim=-1)       # (B, 1+m)
        a_raw = self.mlp(f_in)
        B = a_raw.shape[0]
        a = a_raw.view(B, self.m, self.n_g, self.n_params_per_subfn)

        alpha = a[..., 0]
        beta  = F.softplus(a[..., 1]) + 1e-6
        gamma = a[..., 2]

        t_expand = t.view(B, 1, 1).expand(-1, self.m, self.n_g)  # (B,m,n_g)
        arg = beta * t_expand + gamma
        dot_terms = alpha * beta * torch.cos(arg)   # (B,m,n_g)
        gdot = dot_terms.sum(dim=-1)                # (B,m)
        return gdot

    def f_ssm(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Placeholder state-space model (SSM) for the pendulum and flywheel. 
        The input x has shape (B, m) with m=3 and represents [theta, omega, phi], 
        and u is the scalar torque of shape (B, 1). The output xdot has shape (B, m) and represents 
        [theta_dot, omega_dot, phi_dot]. This is a simple pendulum with damping and a flywheel angle tracked: 
        theta_dot = omega, omega_dot = (u - b * omega - m*g*L*sin(theta)) / I, phi_dot = omega. 
        The constants should be tuned to your real plant and the method should be replaced with 
        the real dynamics of your system."""
        theta = x[:, 0:1]
        omega = x[:, 1:2]
        phi = x[:, 2:3]

        g_const = 9.81
        L = 1.0
        m_pend = 1.0
        I = 1.0     # total inertia about pivot (needs realistic value)
        b = 0.05    # damping

        theta_dot = omega
        omega_dot = (u - b * omega - m_pend * g_const * L * torch.sin(theta)) / I
        phi_dot = omega

        xdot = torch.cat([theta_dot, omega_dot, phi_dot], dim=-1)  # (B,3)
        return xdot

    def compute_loss(self, batch: torch.Tensor, x_data:torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        """Computes the physics loss L_phys = MSE( xdot_hat, f_ssm(x_hat, u(t)) ).
          The batch should have shape (B, 1+1+3) and be in the format [t, u, x0]."""
        x_hat = self.forward(batch)                     # (B, m)
        xdot_hat = self.derivative_closed_form(batch)    # (B, m)

        u = batch[:, 1].unsqueeze(-1)

        f_eval = self.f_ssm(x_hat, u)                   

        mse_data = F.mse_loss(x_hat, x_data, reduction=reduction)

        mse_phy = F.mse_loss(xdot_hat, f_eval, reduction=reduction)

        mse = mse_phy * self.loss_phy_weight + mse_data * (1 - self.loss_phy_weight)

        return mse

    def predict(self, batch: torch.Tensor) -> torch.Tensor:
        """Returns the predicted full state x_hat for the given batch, without gradient calculation."""
        with torch.no_grad():
            return self.forward(batch)


class LightningDDPINN(L.LightningModule):
    def __init__(self, model: Optional[DDPINN] = None, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = model if model is not None else DDPINN()
        self.lr = lr

    def step(self, batch):
        x_state, x_data = batch
        loss = self.model.compute_loss(x_state, x_data, reduction="mean")
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("test/loss", loss)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return opt


# MAIN TEST FUNCTION
def main():
    import numpy as np
    torch.manual_seed(0)
    np.random.seed(0)
    batch_size = 4
    m = 3
    t = np.linspace(0, 1, batch_size).reshape(-1, 1)
    u = np.ones((batch_size, 1)) * 0.5
    theta = np.random.uniform(-np.pi, np.pi, (batch_size, 1))
    omega = np.random.uniform(-1, 1, (batch_size, 1))
    phi = np.random.uniform(-np.pi, np.pi, (batch_size, 1))
    x0 = np.concatenate([theta, omega, phi], axis=1)
    batch_np = np.concatenate([t, u, x0], axis=1)
    batch = torch.tensor(batch_np, dtype=torch.float32)
    x_data = torch.tensor(np.random.uniform(-1, 1, (batch_size, m)), dtype=torch.float32)
    model = DDPINN(m=m)
    x_hat = model.forward(batch)
    print("Wyjście sieci (x_hat):\n", x_hat)
    loss = model.compute_loss(batch, x_data)
    print("Strata (loss):", loss.item())
    x_pred = model.predict(batch)
    print("Predict (bez grad):\n", x_pred)

if __name__ == "__main__":
    main()

