import torch
import torch.nn
import lightning as pl
from data.dataset import PINNDataModule
from pinn.pinc.pinc import RWP_PINC_PL
import argparse
import numpy as np
import math
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import wandb

def run():
    parser = argparse.ArgumentParser(description="Physics-Informed Neural Network for Control (PINC) Trainer")
    # ==================== Model parameters and configs ====================
    parser.add_argument("-cfg", "--model_cfg", type=str, default="config/pinc_model.yaml",
                        help="Path to model architecture config file (.yaml)")

    parser.add_argument("--data_path", type=str, default="data/RWP_SQUARE_EXP.h5",
                        help="Path to the h5 dataset")
    
    parser.add_argument("-Ts", "--sample_time", type=float, default=0.01,
                        help="Sampling time step  of the real experimental dataset (dt) in seconds")

    parser.add_argument("-nx", "--n_states", type=int, default=4,
                        help="Number of state variables (dimension of state space)")

    parser.add_argument("-nu", "--n_control", type=int, default=1,
                        help="Number of control inputs (dimension of control space)")

    parser.add_argument("-T", "--horizon_T", type=float, default=0.5,
                        help="Prediction horizon length in seconds")

    parser.add_argument("--preprocess", type=str, choices=['none', 'rwp_square'],
                        default='rwp_square', help="Data preprocessing method if needed")

    parser.add_argument("-SB", "--State_Boundaries", type=bounds_type, default=((0,math.pi),(-4,4),(-200,200)),
                        help="State boundaries")

    parser.add_argument("-CB", "--Control_Boundaries", type=bounds_type, default=((-0.5,0.5)),
                        help="Control boundaries")

    parser.add_argument("--scale_factors", type=tuple_type, default=(math.pi,4,200),
                        help="Scaling factors")

    # ==================== Data generation hyperparams ====================
    parser.add_argument("--physics_samples", type=int, default=100000,
                        help="Number of physics/collocation points to sample")

    parser.add_argument("--physics_chunk_size", type=int, default=5000,
                        help="Chunk size for generating physics points")

    parser.add_argument("--ic_samples", type=int, default=1000,
                        help="Number of initial condition samples")

    parser.add_argument("--ic_chunk_size", type=int, default=1000,
                        help="Chunk size for initial condition sampling")

    parser.add_argument("--train_split", type=float, default=0.4,
                        help="Train/validation split ratio (0.0 - 1.0) -> train+val+test==1")

    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Validation split ratio -> train+val+test==1")

    parser.add_argument("--pre_generate", type=int, default=1, choices=[0, 1],
                        help="1 = pre-generate all data before training, 0 = on-the-fly")

    # ==================== Trainer hyperparams ====================
    parser.add_argument("--logdir", type=str, default="logs/pinc",
                        help="Directory for logs and checkpoints")

    parser.add_argument("--wandb_name", type=str, default="PINC_Initial_test",
                        help="Name for the current wandb run")
    
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility. -1 = no fixed seed")

    parser.add_argument("-e", "--epochs", type=int, default=1000,
                        help="Number of training epochs")

    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3,
                        help="Initial learning rate")

    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="L2 regularization (weight decay)")

    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")

    parser.add_argument("--num_workers", type=int, default=6,
                        help="Number of DataLoader workers")

    parser.add_argument("--device", type=str, default="cuda",
                        choices=['cpu', 'cuda'], help="Device to train on")

    parser.add_argument("--lambda_ic", type=float, default=1.0,
                        help="Weight for initial condition loss")

    parser.add_argument("--lambda_physics", type=float, default=1e-3,
                        help="Weight for PDE/residual (physics) loss")

    parser.add_argument("--lambda_exp", type=float, default=1.0,
                        help="Weight for experimental/data matching loss (if applicable)")
    
    cli_args = parser.parse_args()
    print(f"Command line arguments: {cli_args}")

    n_states = cli_args.n_states
    n_control = cli_args.n_control
    device = "cpu"
    if cli_args.device == "cuda":
        device = cli_args.device if torch.cuda.is_available() else 'cpu'
    if device not in ["cpu","cuda"]:
        raise Exception(f"Unexpected device type got - {device}")
    
    if cli_args.seed == -1:
        pl.seed_everything(None)
    else:
        pl.seed_everything(cli_args.seed)
    
    data_path = {}
    data_path["train"] = cli_args.data_path
    
    Pinc_DataModule = PINNDataModule(
            data_path=data_path,
            n_states=n_states,
            n_control=n_control,
            horizon_T=cli_args.horizon_T,
            dt= cli_args.sample_time,
            state_boundaries=cli_args.State_Boundaries,
            control_boundaries=cli_args.Control_Boundaries,
            device=device,
            physics_samples = cli_args.physics_samples,
            chunk_size_physics = cli_args.physics_chunk_size,
            ic_samples= cli_args.ic_samples,
            chunk_size_ic = cli_args.ic_chunk_size,
        #  chunk_size_exp:int = 100,
            random_noise=None, 
            train_split =cli_args.train_split,
            val_split =cli_args.val_split,
            batch_size = cli_args.batch_size,
            num_workers = cli_args.num_workers,
            pre_generate = cli_args.pre_generate,
        )

    if cli_args.preprocess == "rwp_square":
        print("Check if the pre_process functions has correct transformations!")
        Pinc_DataModule,n_states,n_control = pre_process_rwp_square(Pinc_DataModule,n_states,n_control)

    Pinc_DataModule.setup()

    model_lightning = RWP_PINC_PL(
                model_descriptor_path=cli_args.model_cfg,
                n_states=n_states,
                n_control=n_control,
                horizon_T=cli_args.horizon_T,
                state_boundaries=cli_args.State_Boundaries,
                control_boundaries=cli_args.Control_Boundaries,
                lambda_ic = cli_args.lambda_ic,
                lambda_physics = cli_args.lambda_physics, 
                lambda_data = cli_args.lambda_exp,
                scale_factors = cli_args.scale_factors,
                lr = cli_args.learning_rate, 
                weight_decay= cli_args.weight_decay
                )

    wandb.login()
    wandb_logger = WandbLogger(project="PINN_RL", 
                           entity="deep-neural-network-course",
                           group="PINC",
                           name=cli_args.wandb_name,#Rename it correctly
                           log_model=True)


    checkpoint_callback = ModelCheckpoint(
        dirpath=cli_args.logdir,
        filename="PINC-{epoch:02d}-{train_loss:.4f}",
        save_top_k=1,
        monitor="train_loss",
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=cli_args.epochs,
        logger=wandb_logger,
        accelerator="gpu" if device == "cuda" else "cpu",
        enable_progress_bar=True,
        accumulate_grad_batches=1,
        callbacks=[checkpoint_callback],
        log_every_n_steps=1,
    )

    trainer.fit(model=model_lightning,
                train_dataloaders=Pinc_DataModule.train_dataloader(), 
                val_dataloaders=Pinc_DataModule.val_dataloader())
    


def pre_process_rwp_square(PDM:PINNDataModule,n_states,n_control):
    ## Clip training samples
    PDM.clip_samples("train",range=(0.125,0.925))
    ## Wrap pendulum angle to [-pi,pi]
    PDM.wrap_state(idx=1,key="train",type="wrapToPi")
    ## Scale state ranges to [-1,1] - state*scale_factor
    PDM.scale_state(idx=1,scale_factor=1/np.pi,key="train")
    PDM.scale_state(idx=2,scale_factor=1/4,key="train")
    PDM.scale_state(idx=4,scale_factor=1/200,key="train")
    ## Remove the DC angle state
    PDM.remove_state(idx=3, key=None)
    return PDM, n_states - 1, n_control


def bounds_type(s):
    return tuple(
        (float(a), float(b))
        for a, b in (p.strip('()').split(',') for p in s.split())
    )


def tuple_type(s: str):
    cleaned = s.strip().replace('(', '').replace(')', '')
    items = [x.strip() for x in cleaned.split(',') if x.strip()]
    
    return tuple(float(x) for x in items)


if __name__ == "__main__":
    run()