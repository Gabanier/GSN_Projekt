import torch.nn as nn
import torch.cuda
from torch.autograd import grad
import lightning as L
from typing import Any, Tuple, List, Mapping, Union, Optional, Dict
# from utils.utils import sequential_from_descriptor
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from pymatreader import read_mat
import matplotlib.pyplot as plt 
from pathlib import Path

#TODO
class PendulumDataset(Dataset):
    ## Assumptions made about the data going in
    ## (t, u_0 u_1 ..u_n, x_0 x_1 .. x_n) -> correct order for the data 
    ## x_{i} should also map to corressponding dx_{i}
    ## dataset should also be structured 
    ## (n_samples, labels)
    def __init__(self, data:np.ndarray, states:int=4, random_noise: Optional[dict] = None) -> None:
        super().__init__()
        if random_noise is None:
            random_noise = {}

        self.n_states:int = states
        self.n_control:int = data.shape[0] - states - 1
        self.random_noise = random_noise
        self.samples:int = data.shape[1]

        self.data:np.ndarray = data
        self.data = self.data # (num_samples,labels)
        
    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index) -> Any:
        if not self.random_noise:
            return self.data[index,:]
        else:
            ##TODO
            return None
        
    
class RWPDataModule(L.LightningDataModule):
    def __init__(self,data_path:dict[str,Union[str, Path]],states:int, 
                 random_noise:Optional[Dict[str,float]]=None,
                 train_split:float = 0.8,
                 val_split:float = 0.1,
                 New_Ts:float = 0.25,
                 batch_size:int = 32,
                 num_workers:int = 6) -> None:
        
        super().__init__()
        self.data: dict[str,np.ndarray] = {}
        for type, d_path in data_path.items():
            h5py_file = h5py.File(d_path, 'r')
            keys = list(h5py_file.keys())
            if len(keys) > 1:
                raise Exception("Provided h5py file has more than one dataset")
            key = keys[0]
            self.data[type] = np.array(h5py_file.get(key)).transpose()
            h5py_file.close()
        
        self.n_states:int = states
        self.n_control:int = self.data["train"].shape[1] - states - 1
        self.random_noise = random_noise
        self.New_Ts = New_Ts
        self.train_split = train_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers

    def preprocess(self,data:np.ndarray,type:str):
        pass


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
            Ts = train[1, 0] - train[0, 0]
            if Ts <= 0:
                raise Exception("Invalid time step in train dataset")
    
            step = max(1, int(round(self.New_Ts / Ts)))
            subsampled = train[::step, :]
            num_samples = subsampled.shape[0]
            if num_samples < 3:
                raise Exception("Subsampled dataset too small to split")
            # Split sequentially: self.train_split , self.val_split, 1 - (test+val)_split
            train_size = int(self.train_split * num_samples)
            valid_size = int(self.val_split * num_samples)
            test_size = num_samples - train_size - valid_size
            train_new = subsampled[:train_size, :]
            valid = subsampled[train_size:train_size + valid_size, :]
            test = subsampled[train_size + valid_size:, :]
            train = train_new  # Update train to the new subsampled train portion
        elif valid and not test:
            test = valid
        elif test and not valid:
            valid = test

        if valid is None:
            raise Exception("Valid dataset not defined")
        if test is None:
            raise Exception("Test dataset not defined")
        
        self.train_ds = PendulumDataset(train, states=self.n_states, random_noise=self.random_noise)
        self.val_ds = PendulumDataset(valid, states=self.n_states, random_noise={})
        self.test_ds = PendulumDataset(test, states=self.n_states, random_noise={})
        self.data = {}
    
    def train_dataloader(self) -> Any:
        return DataLoader(self.train_ds,
                           batch_size=self.batch_size,
                           shuffle=False,
                           num_workers=self.num_workers,
                           pin_memory= True if torch.cuda.is_available() else False)
    
    def val_dataloader(self) -> Any:
        return DataLoader(self.val_ds,
                           batch_size=self.batch_size,
                           shuffle=False,
                           num_workers=self.num_workers,
                           pin_memory= True if torch.cuda.is_available() else False)
    
    def test_dataloader(self) -> Any:
        return DataLoader(self.test_ds,
                           batch_size=self.batch_size,
                           shuffle=False,
                           num_workers=self.num_workers,
                           pin_memory= True if torch.cuda.is_available() else False)
    
    def remove_state(self, idx:int, key:Optional[str]=None):
        if idx <= self.n_control:
            raise Exception("Can't remove control/time signals")
        else:
            self.n_states = self.n_states - 1
            # if key:
            #     self.data[key] = np.delete(self.data[key], idx, axis=1)
            if True:
                for key in self.data.keys():
                    self.data[key] = np.delete(self.data[key], idx, axis=1)

    def scale_state(self,idx:int,key:str, scale_factor:float):
        if idx <= self.n_control:
            raise Exception("Can't scale control/time signals")
        else:
            self.data[key][:,idx] *= scale_factor
   
    def wrap_state(self,idx:int,key:str,type:str="wrapToPi"):
        if idx <= self.n_control:
            raise Exception("Can't scale control/time signals")
        else:
            if type == "wrapToPi":
                self.data[key][:,idx] = RWPDataModule.wrapToPi(self.data[key][:,idx])
            elif type == "wrapTo2Pi":
                self.data[key][:,idx] = RWPDataModule.wrapTo2Pi(self.data[key][:,idx])
            else:
                raise Exception("Unsupported type")
    
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
        angle[q] = RWPDataModule.wrapTo2Pi(angle[q] + np.pi) - np.pi
        return angle
    

def main():
    noise_dict = {}
    data_path = {}
    data_path["train"] = "./data/RWP_SQUARE_EXP.h5"
    New_Ts = 0.2
    RWPDataMod = RWPDataModule(data_path=data_path,states=4,New_Ts=New_Ts)
    RWPDataMod.clip_samples("train",range=(0.125,0.925))
    RWPDataMod.wrap_state(idx=2,key="train")
    RWPDataMod.scale_state(idx=2,scale_factor=1/np.pi,key="train")
    RWPDataMod.scale_state(idx=3,scale_factor=1/4,key="train")
    RWPDataMod.scale_state(idx=5,scale_factor=1/200,key="train")
    RWPDataMod.remove_state(idx=4, key=None)
    RWPDataMod.setup()
    data = RWPDataMod.val_ds.data
    print(data.shape)
    # data = RWPDataMod.test_ds.data
    # data = RWPDataMod.data["train"]

    # f = h5py.File("./data/RWP_SQUARE_EXP.h5",'r')
    # f = h5py.File("./data/RWP_SQUARE_EXP.h5",'r')
    # key = list(f.keys())[0]
    # data: np.ndarray = np.array(f.get(key))
    # f.close()
    # data:np.ndarray = RWP_ds.data[1000:len(RWP_ds),:]
    # data[2:,:] = PendulumDataset.normalize(data[2:,:])

    # nrng = (-1,1)
    # data = nrng[0] + (nrng[1]-nrng[0])*(data-data.min())/(data.max()-data.min())
    # stds = np.ones((6,1))*1e-3
    # stds = stds.reshape(-1,1)
    # stds[0] = 0
    # stds[1] = 0
    # data_noise = data + np.random.normal(loc=0,scale=stds,size=data.shape)
    
    fig,ax = plt.subplots(2,2)
    time = data[:,0] * New_Ts
    ax[0,0].plot(time,data[:,2])
    ax[0,0].set_title("Kąt wahadła - wrapToPi")
    ax[0,1].plot(time,data[:,3])
    ax[0,1].set_title("Prędkość wahadła")
    ax[1,0].plot(time,data[:,4])
    ax[1,0].set_title("Prędkość DC")
    ax[1,1].plot(time,data[:,1])
    ax[1,1].set_title("Sterowanie")
    # ax[1,0].plot(data[:,4])
    # ax[1,0].set_title("Kąt DC - wrapToPi")
    # ax[1,1].plot(data[:,5])
    # ax[1,1].set_title("Prędkość DC")
    plt.show()
    plt.close()
    


if __name__ == "__main__":
    main()

