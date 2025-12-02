import torch
from pinn.pinc.pinc import RWP_PINC
from copy import copy


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
    
    rand_tensor_state = torch.rand(BATCH_SIZE,n_states,dtype=torch.float32)
    rand_tensor_control = torch.rand(BATCH_SIZE,n_control,dtype=torch.float32)
    rand_tensor_time = torch.rand(BATCH_SIZE,1,dtype=torch.float32)
    try:
        full_tensor = torch.cat([rand_tensor_time,rand_tensor_state,rand_tensor_control],dim=1)
        out = model(full_tensor)
        if list(out.shape) != [BATCH_SIZE,n_states]:
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
    try:
        out1 = model.compute_data_loss(x=x,targets=targets)
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
    x = torch.rand(BATCH_SIZE,1+n_states+n_control, dtype=torch.float32)
    try:
        out1 = model.compute_physics_loss(x=x)
        print(f"mse physics shape == {out1.shape}, random_physics_mse = {out1}")
    except Exception as e:
        print(e)
        return False
    return True


def test_ode_rhs() -> bool:
    model_path = "./pinn/pinc/tests/test_model.yaml"
    n_states = 3
    n_control = 1
    T = 0.1
    BATCH_SIZE = 128
    model = RWP_PINC(model_descriptor_path=model_path, n_states=n_states, n_control=n_control, horizon_T=T)
    rand_tensor_state = torch.rand(BATCH_SIZE,n_states,dtype=torch.float32)
    rand_tensor_control = torch.rand(BATCH_SIZE,n_control,dtype=torch.float32)
    try:
        out = model.ode_rhs(x=rand_tensor_state, u=rand_tensor_control)
        if list(out.shape) != [BATCH_SIZE,n_states]:
            return False
    except Exception as e:
        print(e)
        return False
    return True


def main():
    test_init_res = test_init_()
    print(f"test_init_res == {test_init_res}")
    test1_forward_res = test1_forward()
    print(f"test_forward_res == {test1_forward_res}")
    test_predict_res = test_predict()
    print(f"test_predict_res == {test_predict_res}")
    test_compute_data_ic_loss_res = test_compute_data_ic_loss()
    print(f"test_compute_data_ic_loss_res == {test_compute_data_ic_loss_res}")
    test_compute_physics_res = test_compute_physics_loss()
    print(f"test_compute_physics_res == {test_compute_physics_res}")
    test_ode_rhs_res = test_ode_rhs()
    print(f"test_ode_rhs_res == {test_ode_rhs_res}")


if __name__ == "__main__":
    main()