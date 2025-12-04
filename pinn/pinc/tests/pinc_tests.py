import torch
from pinn.pinc.pinc import RWP_PINC, RWP_PINC_PL
from data.dataset import PINNDataModule
import lightning as pl
from copy import copy
import math
import numpy as np


def test_init_() -> bool:
    model_path = "./pinn/pinc/tests/test_model.yaml"
    try:
        RWP_PINC(model_descriptor_path=model_path, n_states=3, n_control=1, horizon_T=0.1)
    except Exception as e:
        print(e)
        return False
    return True


def test1_forward() -> bool:
    model_path = "./pinn/pinc/tests/test_model.yaml"
    n_states = 3
    n_control = 1
    T = 0.1
    BATCH_SIZE = 128
    model = RWP_PINC(model_descriptor_path=model_path, n_states=n_states, n_control=n_control, horizon_T=T)
    for test in range(0,3):
        if test==1:
            rand_tensor= torch.rand(BATCH_SIZE,1+n_states+n_control,dtype=torch.float32)
            expected_shape = [BATCH_SIZE,n_states]
        elif test==2:
            rand_tensor= torch.rand(1+n_states+n_control,dtype=torch.float32)
            expected_shape = [1,n_states]
        else:
            NUM_SAMPLES = 100
            rand_tensor= torch.rand(BATCH_SIZE,NUM_SAMPLES,1+n_states+n_control,dtype=torch.float32)
            expected_shape = [BATCH_SIZE,NUM_SAMPLES,n_states]

        try:
            out = model(rand_tensor)
            if list(out.shape) != expected_shape:
                return False
        except Exception as e:
            print(e)
            return False
    return True


def test_predict() -> bool:
    model_path = "./pinn/pinc/tests/test_model.yaml"
    n_states = 3
    n_control = 1
    T = 0.1
    model = RWP_PINC(model_descriptor_path=model_path, n_states=n_states, n_control=n_control, horizon_T=T)
    rand_tensor_state = torch.rand(1,n_states,dtype=torch.float32)
    rand_tensor_control = torch.rand(1,n_control,dtype=torch.float32)
    try:
        out = model.predict(state=rand_tensor_state, control=rand_tensor_control)
        if list(out.shape) != [n_states]:
            return False
    except Exception as e:
        print(e)
        return False
    return True


def test_compute_data_ic_loss() -> bool:
    model_path = "./pinn/pinc/tests/test_model.yaml"
    n_states = 3
    n_control = 1
    T = 0.1
    BATCH_SIZE = 1
    model = RWP_PINC(model_descriptor_path=model_path, n_states=n_states, n_control=n_control, horizon_T=T)
    x = torch.rand(BATCH_SIZE,1+n_states+n_control, dtype=torch.float32)
    targets =torch.rand(BATCH_SIZE,n_states, dtype=torch.float32)    
    for test in range(0,3):
        if test==1:
            rand_tensor= torch.rand(BATCH_SIZE,1+n_states+n_control,dtype=torch.float32)
            targets =torch.rand(BATCH_SIZE,n_states, dtype=torch.float32)
        elif test==2:
            rand_tensor= torch.rand(1+n_states+n_control,dtype=torch.float32)
            targets =torch.rand(n_states, dtype=torch.float32)
        else:
            NUM_SAMPLES = 100
            rand_tensor= torch.rand(BATCH_SIZE,NUM_SAMPLES,1+n_states+n_control,dtype=torch.float32)
            targets =torch.rand(BATCH_SIZE,NUM_SAMPLES,n_states, dtype=torch.float32)

        try:
            out1 = model.compute_data_loss(x=rand_tensor,targets=targets)
            print(f"mse ic/data shape == {out1.shape}, random_mse ic/data = {out1}")
        except Exception as e:
            print(e)
            return False
    return True


def test_compute_physics_loss() -> bool:
    model_path = "./pinn/pinc/tests/test_model.yaml"
    n_states = 3
    n_control = 1
    T = 0.1
    BATCH_SIZE = 128
    model = RWP_PINC(model_descriptor_path=model_path, n_states=n_states, n_control=n_control, horizon_T=T)
    for test in range(0,3):
        if test==1:
            rand_tensor= torch.rand(BATCH_SIZE,1+n_states+n_control,dtype=torch.float32)
        elif test==2:
            rand_tensor= torch.rand(1+n_states+n_control,dtype=torch.float32)
        else:
            NUM_SAMPLES = 100
            rand_tensor= torch.rand(BATCH_SIZE,NUM_SAMPLES,1+n_states+n_control,dtype=torch.float32)
        try:
            out1 = model.compute_physics_loss(x=rand_tensor)
            print(f"mse physics shape == {out1.shape}, random_physics_mse = {out1}")
        except Exception as e:
            print(e)
            return False
    return True


# def test_ode_rhs(test:int=0) -> bool:
#     model_path = "./pinn/pinc/tests/test_model.yaml"
#     n_states = 3
#     n_control = 1
#     T = 0.1
#     BATCH_SIZE = 1
#     model = RWP_PINC(model_descriptor_path=model_path, n_states=n_states, n_control=n_control, horizon_T=T)
#     for test in range(0,3):
#         if test==1:
#             rand_tensor_state = torch.rand(BATCH_SIZE,n_states,dtype=torch.float32)
#             rand_tensor_control = torch.rand(BATCH_SIZE,n_control,dtype=torch.float32)
#             expected_shape = [BATCH_SIZE,n_states]
#         elif test==2:
#             rand_tensor_state = torch.rand(n_states,dtype=torch.float32)
#             rand_tensor_control = torch.rand(n_control,dtype=torch.float32)
#             expected_shape = [1,n_states]
#         else:
#             NUM_SAMPLES = 100
#             rand_tensor_state = torch.rand(BATCH_SIZE,NUM_SAMPLES,n_states,dtype=torch.float32)
#             rand_tensor_control = torch.rand(BATCH_SIZE,NUM_SAMPLES,n_control,dtype=torch.float32)
#             expected_shape = [BATCH_SIZE,NUM_SAMPLES,n_states]
#         try:
#             model.denormalize()
#             out = model.ode_rhs(x=rand_tensor_state, u=rand_tensor_control)
#             if list(out.shape) != expected_shape:
#                 return False
#         except Exception as e:
#             print(e)
#             return False
#     return True

def test_plPINC_init_():
    model_path = "./pinn/pinc/tests/test_model.yaml"
    n_states = 3
    n_control = 1
    T = 0.5
    try:
        model_lightning = RWP_PINC_PL(
                 model_descriptor_path=model_path,
                 n_states=n_states,
                 n_control=n_control,
                 horizon_T=T,
                 lambda_ic = 1,
                 lambda_physics = 0.001, 
                 lambda_data = 0,
                 scale_factors = (math.pi,4.0,200.0),
                 lr = 1e-3, 
                 weight_decay= 1e-5
                 )
    except Exception as e:
        print(e)
        return False
    return True

def test_plPINC_training():
    ## Assumes dataloader in /data/dataset.py works
    data_file = "./data/RWP_SQUARE_EXP.h5"
    DATA_PATH = {"train":data_file}
    N_STATES = 4
    N_CONTROL = 1
    HORIZON_T = 0.5
    SAMPLE_TIME = 0.01
    BATCH_SIZE = 32
    NUM_WORKERS = 6
    STATE_BOUNDARIES = ((-math.pi,math.pi),(-4,4),(-200,200))
    CONTROL_BOUNDARIES = ((-0.5, 0.5),)

    PDM = PINNDataModule(
                data_path=DATA_PATH,
                n_states=4,
                n_control=N_CONTROL,
                horizon_T=HORIZON_T,
                dt= SAMPLE_TIME,
                state_boundaries=STATE_BOUNDARIES,
                control_boundaries=CONTROL_BOUNDARIES,
                device="cpu",
                physics_samples = int(1e4),
                chunk_size_physics = 100,
                ic_samples= int(1e3),
                chunk_size_ic = 100,
            #  chunk_size_exp:int = 100,
                random_noise=None, 
                train_split =0.4,
                val_split =0.3,
                batch_size = BATCH_SIZE,
                num_workers = NUM_WORKERS,
                pre_generate = True,
            )
    
    PDM.clip_samples("train",range=(0.125,0.925))
    PDM.wrap_state(idx=1,key="train")
    PDM.scale_state(idx=1,scale_factor=1/np.pi,key="train")
    PDM.scale_state(idx=2,scale_factor=1/4,key="train")
    PDM.scale_state(idx=4,scale_factor=1/200,key="train")
    PDM.remove_state(idx=3, key=None)
    PDM.setup()

    model_path = "./pinn/pinc/tests/test_model.yaml"
    n_states = N_STATES - 1
    n_control = N_CONTROL
    model_lightning = RWP_PINC_PL(
                 model_descriptor_path=model_path,
                 n_states=n_states,
                 n_control=n_control,
                 horizon_T=HORIZON_T,
                 lambda_ic = 1,
                 lambda_physics = 0.001, 
                 lambda_data = 0,
                 scale_factors = (math.pi,4.0,200.0),
                 lr = 1e-3, 
                 weight_decay= 1e-5
                 )
    

    trainer = pl.Trainer(
            max_epochs=1,
            log_every_n_steps=1
        )
    
    try:
        trainer.fit(model=model_lightning, train_dataloaders=PDM.train_dataloader(), val_dataloaders=PDM.val_dataloader())
    except Exception as e:
        print(e)
        return False
    return True


def test_pinc():
    print(f"test_init_() == {test_init_()}")
    print(f"test1_forward() == {test1_forward()}")
    print(f"test_predict() == {test_predict()}")
    print(f"test_compute_data_ic_loss() == {test_compute_data_ic_loss()}")
    # print(f"test_ode_rhs() == {test_ode_rhs()}")
    print(f"test_compute_physics_loss() == {test_compute_physics_loss()}")


def test_LightningPINC():
    print(f"test_plPINC_init_() == {test_plPINC_init_()}")
    print(f"test_plPINC_training() == {test_plPINC_training()}")


if __name__ == "__main__":
    test_pinc()
    # test_LightningPINC()