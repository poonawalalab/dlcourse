import pandas as pd
import numpy as np
import string
import random
import torch
from torch.utils.data import DataLoader, Dataset, Subset

class RegressionDataset(Dataset):
    def __init__(self,
                 data_file:str,
                 normalize:bool=False,
                 ):
        super().__init__()
        df_data:pd.DataFrame = pd.read_pickle(data_file)
        target_labels = [c for c in df_data.columns if c.startswith('Target')]
        df_inputs = df_data.drop(target_labels,axis=1)
        df_targets = df_data[target_labels]
        self._tensor_inputs = torch.tensor(df_inputs.values).unsqueeze(1).float()
        self._tensor_targets = torch.tensor(df_targets.values).view(-1,1,1).float()
        if normalize:
            self.normalize(self._tensor_inputs)
            self.normalize(self._tensor_targets)
        
    
    def __len__(self):
        return len(self._tensor_targets)
    
    def __getitem__(self, index):
        inputs = self._tensor_inputs[index]
        targets = self._tensor_targets[index]
        return inputs,targets
    

    def normalize(self,tensor:torch.Tensor) -> None:
        tensor[:] = (tensor-tensor.amin(0))/(tensor.amax(0)-tensor.amin(0))

class MLPDataLoader():
    def __init__(
            self,
            dataset,
            train_split:float = .8,
            batch_size:int = 10,

        ):
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        split = int(len(dataset)*train_split)
        self._batch_size = batch_size
        self._train_idx = indices[:split]
        self._val_idx = indices[split:]
        self._trainset = Subset(dataset,self._train_idx)
        self._valset = Subset(dataset,self._val_idx)

    def train_dataloader(self):
        return DataLoader(
            self._trainset,
            batch_size=self._batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self._valset,
            batch_size=self._batch_size,
            shuffle=False,
        )    
    

def generateFakeData(n_samples:int,
                     n_input_dimensions:int,
                     n_target_dimensions:int,
                     noise_ratio:float=.01):
    
    rng = np.random.default_rng(13)
    ar_inputs = rng.random((n_samples,n_input_dimensions))
    hi_lo_vals = rng.integers(1,high=10,size=(2,n_input_dimensions),endpoint=True)*np.array([[1,-1]]).T
    for i in range(n_input_dimensions):
        ar_inputs[:,i] = ar_inputs[:,i]*(hi_lo_vals[0,i]-hi_lo_vals[1,i])+hi_lo_vals[1,i]
    target_function = rng.random((n_input_dimensions,n_target_dimensions))*10-5
    targets = ar_inputs@target_function
    noise = rng.normal(0,np.std(targets,axis=0)*noise_ratio,[n_samples,n_target_dimensions])
    targets = targets+noise
    input_names = [string.ascii_uppercase[k] for k in range(n_input_dimensions)]
    target_names = [f"Target_{k}" for k in range(n_target_dimensions)]
    df_out = pd.DataFrame(ar_inputs,columns=input_names)
    df_out[target_names] = targets
    df_out.to_pickle('RegressionData.pkl')

if __name__ == "__main__":
    generateFakeData(100,6,1)