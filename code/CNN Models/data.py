import pandas as pd
import numpy as np
import string
import random
import torch
from typing import List, Tuple, Callable, Any
from torchvision.transforms import v2
from os.path import join
from PIL import Image

from torch.utils.data import DataLoader, Dataset, Subset
def normalize_tensor(tensor:torch.Tensor) -> None:
    tensor[:] = (tensor-tensor.amin(0))/(tensor.amax(0)-tensor.amin(0))

class ImageDataset(Dataset):
    def __init__(self,root_dir):
        super().__init__()
        df_exp:pd.DataFrame = pd.read_excel(join(root_dir,'WeldingParameters.xlsx'))
        target_labels = 'Height'
        img_names = df_exp['ImagePath']
        self._img_path = [join(root_dir,'Images',img) for img in img_names.values]
        df_targets = df_exp[target_labels]
        self._tensor_targets = torch.tensor(df_targets.values).view(-1,1).float()

        self._transform: Callable[[Any], torch.Tensor] = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.CenterCrop((480, 480)),
            v2.Resize((128, 128)),
        ]
        )
        
    
    def __len__(self):
        return len(self._tensor_targets)
    
    def __getitem__(self, index) ->Tuple[torch.Tensor,torch.Tensor]:
        img = Image.open(self._img_path[index])
        inputs = self._transform(img)
        targets = self._tensor_targets[index]
        return inputs,targets


class ImageProcessDataset(Dataset):
    def __init__(self,root_dir):
        super().__init__()
        df_exp:pd.DataFrame = pd.read_excel(join(root_dir,'WeldingParameters.xlsx'))
        target_labels = 'Height'
        process_labels = ['Current','Angle','Speed','Time']
        img_names = df_exp['ImagePath']
        self._img_path = [join(root_dir,'Images',img) for img in img_names.values]
        df_targets = df_exp[target_labels]
        df_process = df_exp[process_labels]
        self._tensor_targets = torch.tensor(df_targets.values).view(-1,1).float()
        self._process_inputs = torch.tensor(df_process.values).float()
        normalize_tensor(self._process_inputs)
        normalize_tensor(self._tensor_targets)

        self._transform: Callable[[Any], torch.Tensor] = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.CenterCrop((480, 480)),
            v2.Resize((128, 128)),
        ]
        )
          
    def __len__(self):
        return len(self._tensor_targets)
    
    def __getitem__(self, index) ->Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        img = Image.open(self._img_path[index])
        img_inputs = self._transform(img)
        process_inputs = self._process_inputs[index]
        targets = self._tensor_targets[index]
        return process_inputs,img_inputs,targets


class ImageMaskDataset(Dataset):
    def __init__(self,root_dir):
        super().__init__()
        df_exp:pd.DataFrame = pd.read_excel(join(root_dir,'WeldingParameters.xlsx'))
        img_names = df_exp['ImagePath']
        mask_names = df_exp['LabelPath']
        self._img_path = [join(root_dir,'Images',img) for img in img_names.values]
        self._mask_path = [join(root_dir,'Labels',msk) for msk in mask_names.values]

        self._transform: Callable[[Any], torch.Tensor] = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.CenterCrop((480, 480)),
            v2.Resize((128, 128)),
        ]
        )
        
    
    def __len__(self):
        return len(self._img_path)
    
    def __getitem__(self, index) ->Tuple[torch.Tensor,torch.Tensor]:
        img = Image.open(self._img_path[index])
        mask_np = np.load(self._mask_path[index])
        inputs = self._transform(img)
        mask = self._transform(mask_np.astype(float))
        return inputs,mask
       
class BasicDataLoader():
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