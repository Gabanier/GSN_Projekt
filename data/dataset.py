import torch.nn as nn
import torch.cuda
from torch.autograd import grad
import lightning as pl
from typing import Any, Tuple, Union, Optional, Dict, Mapping, List
from torch.utils.data import Dataset, DataLoader, ConcatDataset, default_collate
import h5py
import numpy as np 
from pathlib import Path
from collections import defaultdict

StateRange = Tuple[Tuple[float, float], ...]


class InitialCondtionsDataset(Dataset):
    ## Assumptions made about the data going in
    ## (sample_size, (t,x_{1..k},u_{1..j})) -> correct order for the data 
    ## x_{k} should also map to corressponding dx_{k}
    ## e.g for a n_state = 2, n_control = 1 system
    ## correct shape would be (n_samples, 1 + n_state + n_control) = (n_samples, 4)

    def __init__(self,
                 state_boundaries: StateRange,
                 control_boundaries: StateRange,
                 ic_samples:int=1000,  # Total number of IC points (pool_size)
                 chunk_size:int=50,        # Chunk size: points per sample (n_samples = ic_samples // n_init)
                 n_states:int=3, 
                 n_control:int=1,
                 horizon_T:float=0.5,
                 pre_generate:bool=True,
                 normalized:bool=True) -> None:
        super().__init__()

        self.n_states:int = n_states
        self.n_control:int = n_control
        self.ic_samples:int = ic_samples          # Total points
        self.chunk_size:int = chunk_size          # Points per chunk/sample -> one batch would hold ()
        self.horizon_T:float = horizon_T
        self.state_range:np.ndarray = np.zeros((self.n_states,2),dtype=np.float32)
        self.control_range:np.ndarray = np.zeros((self.n_control,2),dtype=np.float32)
        if not normalized:
            for state in range(0,self.state_range.shape[0]):
                state_ranges = state_boundaries[state]
                self.state_range[state][0] = state_ranges[0]
                self.state_range[state][1] = state_ranges[1]
            for control in range(0,self.control_range.shape[0]):
                control_ranges = control_boundaries[control]
                self.control_range[control][0] = control_ranges[0]
                self.control_range[control][1] = control_ranges[1]
        else:
            for state in range(0,self.state_range.shape[0]):
                self.state_range[state][0] = -1.0
                self.state_range[state][1] = 1.0
            for control in range(0,self.control_range.shape[0]):
                self.control_range[control][0] = -1.0
                self.control_range[control][1] = 1.0  

        self.num_chunks = self.ic_samples // self.chunk_size 

        if pre_generate:
            # Pre-generate the pool: (ic_samples, 1 + n_states + n_control)
            self.pool = self.random_data_gen_uniform(self.ic_samples, is_IC=True)
        else:
            self.pool = None
        
    def __len__(self) -> int:
        return self.num_chunks

    def __getitem__(self, index:Any, is_IC:bool=True, labels:bool=True, type:str = "IC") -> Any:
        if self.pool is not None:
            # Grab pre-generated IC chunk
            start = index * self.chunk_size
            end = start + self.chunk_size
            ic_chunk = self.pool[start:end]  # (chunk_size, P)
        else:
            # Generate random chunk
            ic_chunk = self.random_data_gen(self.chunk_size, is_IC=is_IC)

        item = {'x': ic_chunk, 'type': type}        
        if labels:
            # Extract targets: y0 = ic_chunk[:, 1:1+n_states]
            item["targets"] = ic_chunk[:, 1:1 + self.n_states] 
        # else:
        #     item["targets"] = False
  
        return item

    
    def random_data_gen(self, num_elems:int, is_IC:bool=True) -> torch.Tensor:
        ic_chunk = torch.zeros((num_elems, 1 + self.n_states + self.n_control), dtype=torch.float32)
        if is_IC:
            ic_chunk[:, 0] = 0.0  # t=0
        #if not IC we also want to generate random time in range [0,T]
        else:
            ic_chunk[:, 0] = torch.rand(num_elems).float() * self.horizon_T
        for i in range(self.n_states):
            ic_chunk[:, 1 + i] = torch.rand(num_elems).float() * (
                self.state_range[i, 1] - self.state_range[i, 0]
            ) + self.state_range[i, 0]
        for i in range(self.n_control):
            ic_chunk[:, 1 + self.n_states + i] = torch.rand(num_elems).float() * (
                self.control_range[i, 1] - self.control_range[i, 0]
            ) + self.control_range[i, 0]
        return ic_chunk
    
    def random_data_gen_uniform(self, num_elems:int, is_IC:bool=True) -> torch.Tensor:
        pool = np.zeros((num_elems, 1 + self.n_states + self.n_control), dtype=np.float32)
        if is_IC:
            pool[:, 0] = 0.0  # t=0 for all ICs
        else:
            pool[:, 0] = np.random.uniform(0,self.horizon_T, size=num_elems)
        #get random initial conditons y(0)
        for i in range(self.n_states):
            pool[:, 1 + i] = np.random.uniform(
                self.state_range[i, 0], self.state_range[i, 1], size=num_elems
            )
        #get random control in range
        for i in range(self.n_control):
            pool[:, 1 + self.n_states + i] = np.random.uniform(
                self.control_range[i, 0], self.control_range[i, 1], size=num_elems
            )
        return torch.from_numpy(pool).float()


class CollocationPointsDataset(InitialCondtionsDataset):
    def __init__(self, state_boundaries: StateRange, 
                 control_boundaries: StateRange, 
                 ic_samples: int = 1000, 
                 chunk_size: int = 50, 
                 n_states: int = 3, 
                 n_control: int = 1, 
                 horizon_T: float = 0.5, 
                 pre_generate: bool = True,
                 normalized: bool = True) -> None:
        super().__init__(state_boundaries, control_boundaries, ic_samples, chunk_size, n_states, n_control, horizon_T, pre_generate=False, normalized=normalized)

        if pre_generate:
            # Pre-generate the pool: (collocation_samples, 1 + n_states + n_control)
            self.pool = self.random_data_gen_uniform(self.ic_samples, is_IC=False)
        else:
            self.pool = None

    def __len__(self) -> int:
        return super().__len__()
    
    def __getitem__(self, index: Any, is_IC: bool = False, labels: bool = False, type: str = "Collocation") -> Any:
        return super().__getitem__(index, is_IC, labels, type)


class ExperimentDataDataset(Dataset):
    def __init__(self,
                 data_tensor: torch.Tensor,  
                 n_states: int,             
                 n_control: int,            
                 horizon_T: float,
                 dt: float, 
                 ) -> None:
        super().__init__()
        self.n_states = n_states
        self.n_control = n_control
        self.data_tensor = data_tensor
        self.horizon_T = horizon_T
        self.dt = dt
        total_size = data_tensor.shape[0]

        # print(f"co {data_tensor.shape}")
        self.times = data_tensor[:, 0]      # (total_size,)
        self.y_data = data_tensor[:, 1:1 + n_states]  # (total_size, n_states)
        self.u_data = data_tensor[:, 1 + n_states:]   # (total_size, n_control)

        self.steps_per_segment = int(self.horizon_T / self.dt)  # Approximate steps per segment
        if self.steps_per_segment < 1:
            raise ValueError(f"Ts={self.dt} > T={self.horizon_T}")

        self.m = total_size // self.steps_per_segment # Total number of chuncks

        self.segment_starts = torch.arange(0, self.m * self.steps_per_segment, self.steps_per_segment)
        self.time_t = torch.arange(0, self.horizon_T, self.dt).view(-1,1)

    def __len__(self) -> int:
        return self.m #return total number of segments

    def __getitem__(self, idx: int) -> dict:
        if idx >= self.m:
            raise IndexError(f"Index {idx} out of range for {self.m} segments")

        start_idx = self.segment_starts[idx].item()
        end_idx = min(start_idx + self.steps_per_segment, len(self.y_data))  

        # Constant u:
        # set it as the mean average of the control values on the span [0,T]
        u = self.u_data[start_idx:end_idx].mean(dim=0).unsqueeze(0).repeat(self.steps_per_segment,1)  # (n_control,)
        ## or as the first value
        # u = self.u_data[start_idx,:].unsqueeze(0).repeat(self.steps_per_segment,1)  

        # Initial state y(0) at start of segment
        y0 = self.y_data[start_idx,:].clone().unsqueeze(0).repeat(self.steps_per_segment,1)  # (n_states,)
        # print(y0.shape, u.shape, self.time_t.shape, self.u_data.shape, self.n_control, self.n_states)
        x = torch.cat([self.time_t,y0,u],dim=1)  # (n_samples, 1 + n_states + n_control)
        target = self.y_data[start_idx:end_idx,:]

        return {
            'x': x,              # Input for model: (n_samples, 1 + n_states + n_control); [t, y(0), u] per point
            'targets': target,  # Target: (n_samples, n_states) y(t)
            'type': "ExperimentData"
        }


class PINNDataModule(pl.LightningDataModule):
    def __init__(self,data_path:Mapping[str,Union[str, Path]],
                 n_states:int,
                 n_control:int,
                 horizon_T:float,
                 dt:float,
                 state_boundaries: StateRange,
                 control_boundaries:StateRange,
                 device:str,
                 physics_samples:int = int(1e4),
                 chunk_size_physics:int = 100,
                 ic_samples:int = int(1e3),
                 chunk_size_ic:int = 100,
                #  chunk_size_exp:int = 100,
                 random_noise:Optional[Dict[str,float]]=None, 
                 train_split:float = 0.4,
                 val_split:float = 0.3,
                 batch_size:int = 32,
                 num_workers:int = 6,
                 pre_generate:bool = True,
                 tyu:bool = False
                ) -> None:
        
        super().__init__()
        self.data: dict[str,np.ndarray] = {}
        self.n_states: dict[str,int] = {}
        self.n_control: dict[str,int] = {}
        for type, d_path in data_path.items():
            h5py_file = h5py.File(d_path, 'r')
            keys = list(h5py_file.keys())
            if len(keys) > 1:
                raise Exception("Provided h5py file has more than one dataset")
            key = keys[0]
            self.data[type] = np.array(h5py_file.get(key)).transpose()
            if not tyu:
                self.data[type] = tuy_to_tyu(self.data[type],n_control=n_control)
            self.n_states[type] = n_states
            self.n_control[type] = n_control
            h5py_file.close()
        
        self.state_boundaries: StateRange = state_boundaries
        self.control_boundaries: StateRange = control_boundaries
        self.horizon_T:float = horizon_T
        self.dt:float = dt
        self.physics_samples:int = physics_samples
        self.ic_samples:int = ic_samples
        self.chunk_size_physics:int = chunk_size_physics
        self.chunk_size_ic:int = chunk_size_ic
        # self.chunk_size_exp:int = chunk_size_exp
        self.random_noise = random_noise
        self.train_split:float = train_split
        self.val_split:float = val_split
        self.batch_size:int = batch_size
        self.num_workers:int = num_workers
        self.device:str = device
        self.pre_generate:bool = pre_generate
        self.normalized:bool = False

    def setup(self, stage: Optional[str] = None) -> None:
        train = self.data.get("train")
        valid = self.data.get("valid")
        test = self.data.get("test")
        if train is None:
            raise Exception("Train data set not specified")
        elif valid is None and test is None:
            # Assume train is a numpy array of shape (num_samples, num_features)
            # First column is time t, assuming sorted and uniform sampling
            if train.shape[0] < 2:
                raise Exception("Train dataset too small to split")

            num_samples = train.shape[0]
            if num_samples < 3:
                raise Exception("Subsampled dataset too small to split")
            # Split sequentially: self.train_split , self.val_split, 1 - (test+val)_split
            train_size = int(self.train_split * num_samples)
            valid_size = int(self.val_split * num_samples)
            train_new = train[:train_size, :]
            valid = train[train_size:train_size + valid_size, :]
            test = train[train_size + valid_size:, :]
            train = train_new

            #create missing keys for state and control size
            self.n_states["valid"] = self.n_states["train"]
            self.n_control["valid"] = self.n_control["train"] 
            self.n_states["test"] = self.n_states["train"]
            self.n_control["test"] = self.n_control["train"] 
        elif valid and not test:
            test = valid
            self.n_states["test"] = self.n_states["valid"]
            self.n_control["test"] = self.n_control["valid"] 
        elif test and not valid:
            valid = test
            self.n_states["valid"] = self.n_states["test"]
            self.n_control["valid"] = self.n_control["test"] 

        if valid is None:
            raise Exception("Valid dataset not defined")
        if test is None:
            raise Exception("Test dataset not defined")
        
        # Convert data to torch tensors
        train_data_torch = torch.from_numpy(train).float()
        valid_data_torch = torch.from_numpy(valid).float()
        test_data_torch = torch.from_numpy(test).float()
        
        # Apply random noise if specified
        if self.random_noise:
            for key, noise_std in self.random_noise.items():
                if key in ['train', 'valid', 'test']:
                    noise = torch.normal(0, noise_std, size=locals()[f"{key}_data_torch"].shape)
                    locals()[f"{key}_data_torch"] += noise
        
        # Experiment datasets
        self.exp_train_ds = ExperimentDataDataset(train_data_torch, 
                                                  self.n_states["train"], 
                                                  self.n_control["train"], 
                                                  self.horizon_T,self.dt)
        self.exp_val_ds = ExperimentDataDataset(valid_data_torch, 
                                                self.n_states["valid"], 
                                                self.n_control["valid"], 
                                                self.horizon_T,self.dt)
        self.exp_test_ds = ExperimentDataDataset(test_data_torch, 
                                                 self.n_states["test"], 
                                                 self.n_control["test"], 
                                                 self.horizon_T,self.dt)
        
        # Physics-informed datasets for training only
        self.ic_train_ds = InitialCondtionsDataset(self.state_boundaries, 
                                                   self.control_boundaries, 
                                                   self.ic_samples, self.chunk_size_ic, 
                                                   self.n_states["train"], 
                                                   self.n_control["train"], 
                                                   self.horizon_T, 
                                                   pre_generate=self.pre_generate,
                                                   normalized=True)
        self.coll_train_ds = CollocationPointsDataset(self.state_boundaries, 
                                                      self.control_boundaries, 
                                                      self.physics_samples, 
                                                      self.chunk_size_physics, 
                                                      self.n_states["train"], 
                                                      self.n_control["train"], 
                                                      self.horizon_T, 
                                                      pre_generate=self.pre_generate,
                                                      normalized=True)
        
        # Val and test: only experiment data
        self.val_ds = self.exp_val_ds
        self.test_ds = self.exp_test_ds
        
        # Clear loaded data to save memory
        self.data = {}
    

    def train_dataloader(self) -> List[DataLoader]:
        return [DataLoader(self.exp_train_ds, batch_size=self.batch_size, shuffle=True, 
                          num_workers=self.num_workers,
                          pin_memory=True if self.device == "cuda" else False,),
                DataLoader(self.ic_train_ds, batch_size=self.batch_size, shuffle=True, 
                          num_workers=self.num_workers,
                          pin_memory=True if self.device == "cuda" else False,),
                DataLoader(self.coll_train_ds, batch_size=self.batch_size, shuffle=True, 
                          num_workers=self.num_workers,
                          pin_memory=True if self.device == "cuda" else False,)]
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, 
                          num_workers=self.num_workers,
                          pin_memory=True if self.device == "cuda" else False,)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, 
                          num_workers=self.num_workers,
                          pin_memory=True if self.device == "cuda" else False,)
    

    def remove_state(self, idx:int, key:Optional[str]=None) -> None:
        if idx < 1:
            raise Exception("Can't remove time signals")
        else:
            if key:
                self.data[key] = np.delete(self.data[key], idx, axis=1)
                self._remove_state_helper(idx=idx, key=key)
            else:
                for key in self.data.keys():
                    self.data[key] = np.delete(self.data[key], idx, axis=1)
                    self._remove_state_helper(idx=idx, key=key)


    def _remove_state_helper(self, idx:int, key:str) -> None:
        if idx > self.n_states[key]:
            self.n_control[key] = self.n_control[key] - 1
        else:
            self.n_states[key] = self.n_states[key] - 1


    def scale_state(self,idx:int,key:str, scale_factor:float):
        if idx < 1:
            raise Exception("Can't scale time signals")
        else:
            self.data[key][:,idx] *= scale_factor
   

    def wrap_state(self,idx:int,key:str,type:str="wrapToPi"):
        if idx < 1:
            raise Exception("Can't scale time signals")
        else:
            if type == "wrapToPi":
                self.data[key][:,idx] = PINNDataModule.wrapToPi(self.data[key][:,idx])
            elif type == "wrapTo2Pi":
                self.data[key][:,idx] = PINNDataModule.wrapTo2Pi(self.data[key][:,idx])
            else:
                raise Exception("Unsupported type")
    
    def normalize_states(self,key:str):
        ##assumes tyu format
        i, j = 0, 0
        for idx in range(1,1+self.n_states[key]):
            nrng = self.state_boundaries[i]
            i += 1
            self.data[key][:,idx] = PINNDataModule.normalize(self.data[key][:,idx],nrng=nrng)
        for idx in range(1+self.n_states[key],1+self.n_states[key]+self.n_control[key]):
            nrng = self.control_boundaries[j]
            j += 1 
            self.data[key][:,idx] = PINNDataModule.normalize(self.data[key][:,idx],nrng=nrng) 
    
    def clip_samples(self,type, range:tuple[float,float]=(0.,1.)):
        num_samples = self.data[type][:,0].size
        beg_idx = int(range[0]*num_samples)
        end_idx = int(range[1]*num_samples)
        self.data[type] = self.data[type][beg_idx:end_idx, :]

    @staticmethod
    def normalize(data: np.ndarray, nrng:Tuple[float,float] = (-1.,1.)) -> np.ndarray:
        data_shape = data.shape
        if len(data_shape) == 1:
            data = data.reshape(1,-1)
        elif len(data_shape) > 2:
            raise Exception("Function allows only for arrays with 1 or 2 axis")
        data_min = data.min(axis=0,keepdims=True)
        data_max = data.max(axis=0,keepdims=True)
        return nrng[0] + (nrng[1]-nrng[0])*(data-data_min)/(data_max-data_min+1e-12)
    
    @staticmethod
    def wrapTo2Pi(angle: np.ndarray) -> np.ndarray:
        positiveInput = (angle > 0)
        angle = np.mod(angle, 2 * np.pi)
        angle[(angle == 0) & positiveInput] = 2 * np.pi
        return angle

    @staticmethod
    def wrapToPi(angle: np.ndarray) -> np.ndarray:
        q = (angle < -np.pi) | (angle > np.pi)
        angle[q] = PINNDataModule.wrapTo2Pi(angle[q] + np.pi) - np.pi
        return angle


def tuy_to_tyu(data_tensor:np.ndarray, n_control:int) -> np.ndarray:
    times = data_tensor[:, 0:1]
    u_block = data_tensor[:, 1:1 + n_control]
    y_block = data_tensor[:, 1 + n_control:]
    reversed_tensor = np.concatenate([times, y_block, u_block], axis=1)
    return reversed_tensor

def tyu_to_tuy(data_tensor:np.ndarray, n_control:int) -> np.ndarray:
    samples, labels = data_tensor.shape
    times = data_tensor[:, 0:1]
    u_block = data_tensor[:, labels-n_control:]
    y_block = data_tensor[:, 1:labels-n_control+1]
    reversed_tensor = np.concatenate([times, u_block, y_block], axis=1)
    return reversed_tensor
