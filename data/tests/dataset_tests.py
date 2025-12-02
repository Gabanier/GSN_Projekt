import data.dataset as dst
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt 
import math

N_STATES = 4
N_CONTROL = 1
HORIZON_T = 0.5
SAMPLE_TIME = 0.01
BATCH_SIZE = 32
NUM_WORKERS = 6
Ts = 0.01
STATE_BOUNDARIES = ((-math.pi,math.pi),(-4,4),(-200,200))
CONTROL_BOUNDARIES = ((-0.5, 0.5),)
data_file = "./data/RWP_SQUARE_EXP.h5"
DATA_PATH = {"train":data_file}

f = h5py.File(data_file,'r')
key = list(f.keys())[0]
DATA_NUMPY: np.ndarray = np.array(f.get(key)).transpose()
f.close()
DATA_NUMPY = np.delete(DATA_NUMPY,4,axis=1)
N_STATES -= 1
print(f"Data shape = {DATA_NUMPY.shape}")
DATA_NUMPY = dst.tuy_to_tyu(DATA_NUMPY,n_control=N_CONTROL)

# fig,ax = plt.subplots(2,2)
# time = DATA_NUMPY[:,0]
# control = DATA_NUMPY[:,5]
# state_1 = DATA_NUMPY[:,1]
# state_2 = DATA_NUMPY[:,2]
# state_3 = DATA_NUMPY[:,3]
# state_4 = DATA_NUMPY[:,4]
# print(time.max(), time.min())
# ax[0,0].plot(time,state_1)
# ax[0,0].set_title("Kąt wahadła")
# ax[0,1].plot(time,state_2)
# ax[0,1].set_title("Prędkość wahadła")
# ax[1,0].plot(time,state_3)
# ax[1,0].set_title("Kąt DC")
# ax[1,1].plot(time,state_4)
# ax[1,1].set_title("Prędkość DC")
# plt.show()
# plt.close()

DATA_TENSOR = torch.from_numpy(DATA_NUMPY).float()

def check_boundaries(tensor: torch.Tensor, state_boundaries: tuple, control_boundaries: tuple):
    # Unpack state mins and maxs
    state_mins = torch.tensor([b[0] for b in state_boundaries])
    state_maxs = torch.tensor([b[1] for b in state_boundaries])
    
    # Unpack control mins and maxs
    control_mins = torch.tensor([b[0] for b in control_boundaries])
    control_maxs = torch.tensor([b[1] for b in control_boundaries])
    
    # Extract states and controls
    states = tensor[:, 1:4]  # (N, 3)
    controls = tensor[:, 4:5]  # (N, 1)
    
    # Check states: all within per-state bounds
    state_ok = torch.all(
        (states >= state_mins.unsqueeze(0)) & (states <= state_maxs.unsqueeze(0))
    )
    
    # Check controls
    control_ok = torch.all(
        (controls >= control_mins.unsqueeze(0)) & (controls <= control_maxs.unsqueeze(0))
    )
    
    return state_ok and control_ok

def test_ExperimentData_init_():
    try:
        Exp_Dataset = dst.ExperimentDataDataset(
                    data_tensor=DATA_TENSOR, 
                    dt=Ts, 
                    n_states=N_STATES,             
                    n_control=N_CONTROL,            
                    horizon_T=HORIZON_T, 
                    ) 
    except Exception as e:
        print(e)
        return False
    return True

def test_ExperimentData_len():
    Exp_Dataset = dst.ExperimentDataDataset(
            data_tensor=DATA_TENSOR,  
            dt=Ts,
            n_states=N_STATES,             
            n_control=N_CONTROL,            
            horizon_T=HORIZON_T, 
            ) 
    try:
        print(f"dataset len == {len(Exp_Dataset)}")
    except Exception as e:
        print(e)
        return False
    return True

def test_ExperimentData_get():
    Exp_Dataset = dst.ExperimentDataDataset(
            data_tensor=DATA_TENSOR, 
            dt=Ts, 
            n_states=N_STATES,             
            n_control=N_CONTROL,            
            horizon_T=HORIZON_T, 
            )
    try:
        item = Exp_Dataset[101]
        print(f"item keys == {item.keys()}")
        print(f"x = {item['x'].shape}, target = {item['targets'].shape}")
    except Exception as e:
        print(e)
        return False
    
    data = item['targets'].detach().numpy()
    fig,ax = plt.subplots(2,2)
    time = item['x'][:,0]
    state_1 = data[:,0]
    state_2 = data[:,1]
    state_3 = data[:,2]
    print(time.max(), time.min())
    ax[0,0].plot(time,state_1)
    ax[0,0].set_title("Kąt wahadła")
    ax[0,1].plot(time,state_2)
    ax[0,1].set_title("Prędkość wahadła")
    ax[1,1].plot(time,state_3)
    ax[1,1].set_title("Prędkość DC")
    plt.show()
    plt.close()
    return True


def test_ICDataset_init():
    try:
        IC_Dataset = dst.InitialCondtionsDataset(
                    state_boundaries=STATE_BOUNDARIES,
                    control_boundaries=CONTROL_BOUNDARIES,
                    ic_samples=1000,
                    chunk_size=50,  
                    n_states=N_STATES,             
                    n_control=N_CONTROL,            
                    horizon_T=HORIZON_T,
                    pre_generate=False 
                    )
        
        IC_Dataset = dst.InitialCondtionsDataset(
                    state_boundaries=STATE_BOUNDARIES,
                    control_boundaries=CONTROL_BOUNDARIES,
                    ic_samples=1000,
                    chunk_size=50,  
                    n_states=N_STATES,             
                    n_control=N_CONTROL,            
                    horizon_T=HORIZON_T,
                    pre_generate=True 
                    ) 
    except Exception as e:
        print(e)
        return False
    return True


def test_ICDataset_len():
    IC_Dataset_1 = dst.InitialCondtionsDataset(
                state_boundaries=STATE_BOUNDARIES,
                control_boundaries=CONTROL_BOUNDARIES,
                ic_samples=1000,
                chunk_size=50,  
                n_states=N_STATES,             
                n_control=N_CONTROL,            
                horizon_T=HORIZON_T,
                pre_generate=False 
                )
    
    IC_Dataset_2 = dst.InitialCondtionsDataset(
                state_boundaries=STATE_BOUNDARIES,
                control_boundaries=CONTROL_BOUNDARIES,
                ic_samples=1000,
                chunk_size=50,  
                n_states=N_STATES,             
                n_control=N_CONTROL,            
                horizon_T=HORIZON_T,
                pre_generate=True 
                ) 
    try:
        print(f"IC1 len == {len(IC_Dataset_1)}")
        print(f"IC2 len == {len(IC_Dataset_2)}")
    except Exception as e:
        print(e)
        return False
    return True


def test_ICDataset_get():
    IC_Dataset_1 = dst.InitialCondtionsDataset(
                state_boundaries=STATE_BOUNDARIES,
                control_boundaries=CONTROL_BOUNDARIES,
                ic_samples=1000,
                chunk_size=50,  
                n_states=N_STATES,             
                n_control=N_CONTROL,            
                horizon_T=HORIZON_T,
                pre_generate=False 
                )
    
    IC_Dataset_2 = dst.InitialCondtionsDataset(
                state_boundaries=STATE_BOUNDARIES,
                control_boundaries=CONTROL_BOUNDARIES,
                ic_samples=1000,
                chunk_size=50,  
                n_states=N_STATES,             
                n_control=N_CONTROL,            
                horizon_T=HORIZON_T,
                pre_generate=True 
                ) 
    try:
        item1 = IC_Dataset_1[0]
        item2 = IC_Dataset_2[1]
        print(f"IC1 shape == {item1['x'].shape}, {item1.keys()}")
        print(f"within boundaries == {check_boundaries(item1['x'],STATE_BOUNDARIES,CONTROL_BOUNDARIES)}")
        print(f"item1 max = {torch.max(item1['x'],dim=0).values}")
        print(f"item1 min = {torch.min(item1['x'],dim=0).values}")
        print(f"mean of target - y0 = {(item1['x'][:,1:4]-item1['targets']).mean()}")
        print(f"IC2 shape == {item2['x'].shape}, {item2.keys()}")
        print(f"within boundaries == {check_boundaries(item2['x'],STATE_BOUNDARIES,CONTROL_BOUNDARIES)}")
        print(f"item2 max = {torch.max(item2['x'],dim=0).values}")
        print(f"item2 min = {torch.min(item2['x'],dim=0).values}")
        print(f"mean of target - y0 = {(item2['x'][:,1:4]-item2['targets']).mean()}")
    except Exception as e:
        print(e)
        return False
    return True


def test_CollcDataset_init():
    try:
        Collc_Dataset = dst.CollocationPointsDataset(
                    state_boundaries=STATE_BOUNDARIES,
                    control_boundaries=CONTROL_BOUNDARIES,
                    ic_samples=1000,
                    chunk_size=50,  
                    n_states=N_STATES,             
                    n_control=N_CONTROL,            
                    horizon_T=HORIZON_T,
                    pre_generate=False 
                    )
        
        Collc_Dataset = dst.CollocationPointsDataset(
                    state_boundaries=STATE_BOUNDARIES,
                    control_boundaries=CONTROL_BOUNDARIES,
                    ic_samples=1000,
                    chunk_size=50,  
                    n_states=N_STATES,             
                    n_control=N_CONTROL,            
                    horizon_T=HORIZON_T,
                    pre_generate=True 
                    ) 
    except Exception as e:
        print(e)
        return False
    return True


def test_CollcDataset_len():
    Collc_Dataset_1 = dst.CollocationPointsDataset(
                state_boundaries=STATE_BOUNDARIES,
                control_boundaries=CONTROL_BOUNDARIES,
                ic_samples=1000,
                chunk_size=50,  
                n_states=N_STATES,             
                n_control=N_CONTROL,            
                horizon_T=HORIZON_T,
                pre_generate=False 
                )
    
    Collc_Dataset_2 = dst.CollocationPointsDataset(
                state_boundaries=STATE_BOUNDARIES,
                control_boundaries=CONTROL_BOUNDARIES,
                ic_samples=1000,
                chunk_size=50,  
                n_states=N_STATES,             
                n_control=N_CONTROL,            
                horizon_T=HORIZON_T,
                pre_generate=True 
                ) 
    try:
        print(f"IC1 len == {len(Collc_Dataset_1)}")
        print(f"IC2 len == {len(Collc_Dataset_2)}")
    except Exception as e:
        print(e)
        return False
    return True


def test_CollcDataset_get():
    Collc_Dataset_1 = dst.CollocationPointsDataset(
                state_boundaries=STATE_BOUNDARIES,
                control_boundaries=CONTROL_BOUNDARIES,
                ic_samples=1000,
                chunk_size=50,  
                n_states=N_STATES,             
                n_control=N_CONTROL,            
                horizon_T=HORIZON_T,
                pre_generate=False 
                )
    
    Collc_Dataset_2 = dst.CollocationPointsDataset(
                state_boundaries=STATE_BOUNDARIES,
                control_boundaries=CONTROL_BOUNDARIES,
                ic_samples=1000,
                chunk_size=50,  
                n_states=N_STATES,             
                n_control=N_CONTROL,            
                horizon_T=HORIZON_T,
                pre_generate=True 
                ) 
    try:
        item1 = Collc_Dataset_1[0]
        item2 = Collc_Dataset_2[1]
        print(f"Collc1 shape == {item1['x'].shape}, {item1.keys()}")
        print(f"within boundaries == {check_boundaries(item1['x'],STATE_BOUNDARIES,CONTROL_BOUNDARIES)}")
        print(f"item1 max = {torch.max(item1['x'],dim=0).values}")
        print(f"item1 min = {torch.min(item1['x'],dim=0).values}")
        print(f"target1  {item1['targets']}")
        print(f"Collc2 shape == {item2['x'].shape}, {item2.keys()}")
        print(f"within boundaries == {check_boundaries(item2['x'],STATE_BOUNDARIES,CONTROL_BOUNDARIES)}")
        print(f"item2 max = {torch.max(item2['x'],dim=0).values}")
        print(f"item2 min = {torch.min(item2['x'],dim=0).values}")
        print(f"target2  {item2['targets']}")
    except Exception as e:
        print(e)
        return False
    return True


def test_PINNDataModule_init():
    try:
        PDM = dst.PINNDataModule(
                 data_path=DATA_PATH,
                 n_states=N_STATES,
                 n_control=N_CONTROL,
                 horizon_T=HORIZON_T,
                 dt = Ts,
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
    except Exception as e:
        print(e)
        return False
    return True

def test_PINNDataModule_setup():
    PDM = dst.PINNDataModule(
                data_path=DATA_PATH,
                n_states=4,
                n_control=N_CONTROL,
                horizon_T=HORIZON_T,
                dt = Ts,
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
    try:
        PDM.clip_samples("train",range=(0.125,0.925))
        PDM.wrap_state(idx=1,key="train")
        PDM.scale_state(idx=1,scale_factor=1/np.pi,key="train")
        PDM.scale_state(idx=2,scale_factor=1/4,key="train")
        PDM.scale_state(idx=4,scale_factor=1/200,key="train")
        print(PDM.data["train"].shape)
        print(f"before n_control {PDM.n_control['train']}")
        print(f"before n_states {PDM.n_states['train']}")
        PDM.remove_state(idx=3, key=None)
        print(PDM.data["train"].shape)
        print(f"after n_control {PDM.n_control['train']}")
        print(f"after n_states {PDM.n_states['train']}")
        PDM.setup()
    except Exception as e:
        print(e)
        return False
    return True

def test_PINNDataModule_dataloaders():
    PDM = dst.PINNDataModule(
                data_path=DATA_PATH,
                n_states=4,
                n_control=N_CONTROL,
                horizon_T=HORIZON_T,
                dt= Ts,
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
    try:
        test_dataloader = PDM.test_dataloader()
        dict_x = next(iter(test_dataloader))
        x = dict_x['x']
        typ = dict_x["type"]
        targets = dict_x['targets']
        print("test dataloader")
        print(x.shape, typ[0] , targets.shape)

        val_dataloader = PDM.val_dataloader()
        dict_x = next(iter(val_dataloader))
        x = dict_x['x']
        typ = dict_x["type"]
        targets = dict_x['targets']
        print("valid dataloader")
        print(x.shape, typ[0] , targets.shape)

        #train dataloaders
        train_exp_dl, train_ic_dl, train_coll_dl = PDM.train_dataloader()
        dict_x = next(iter(train_exp_dl))
        x = dict_x['x']
        typ = dict_x["type"]
        targets = dict_x['targets']
        print("train dataloader EXP data")
        print(dict_x.keys())
        print(x.shape, typ[0] , targets.shape)

        dict_x = next(iter(train_ic_dl))
        x = dict_x['x']
        typ = dict_x["type"]
        targets = dict_x['targets']
        print("train dataloader IC")
        print(dict_x.keys())
        print(x.shape, typ[0] , targets.shape)

        dict_x = next(iter(train_coll_dl))
        x = dict_x['x']
        typ = dict_x["type"]
        
        print("train dataloader collocation")
        print(dict_x.keys())
        print(x.shape, typ[0])
    except Exception as e:
        print(e)
        return False
    return True

def test_ExperimentData():
    print(f"test_ExperimentData_init_() = {test_ExperimentData_init_()}")
    print(f"test_ExperimentData_len() = {test_ExperimentData_len()}")
    print(f"test_ExperimentData_get() = {test_ExperimentData_get()}")

def test_InitialCondtionsDataset():
    print(f"test_ICDataset_init() = {test_ICDataset_init()}")
    print(f"test_ICDataset_len() = {test_ICDataset_len()}")
    print(f"test_ICDataset_get() = {test_ICDataset_get()}")

def test_CollocationPointsDataset():
    print(f"test_CollcDataset_init() = {test_CollcDataset_init()}")
    print(f"test_CollcDataset_len() = {test_CollcDataset_len()}")
    print(f"test_CollcDataset_get() = {test_CollcDataset_get()}")

def test_PINNDataModule():
    print(f"test_PINNDataModule_init() = {test_CollcDataset_init()}")
    print(f"test_PINNDataModule_setup() = {test_PINNDataModule_setup()}")
    print(f"test_PINNDataModule_dataloaders() = {test_PINNDataModule_dataloaders()}")

if __name__ == "__main__":
    # test_ExperimentData()
    # print()
    # test_InitialCondtionsDataset()
    # print()
    # test_CollocationPointsDataset()
    print()
    test_PINNDataModule()