from typing import List
import torch
import torch.nn as nn
from utils import list_check

class mlpModel(nn.Module):
    def __init__(self,
                 in_neurons:int,
                 hidden_neurons:int|List[int],
                 out_neurons:int,
                 dropout:float = 0.,
                )->None:
        
        super().__init__()
        hidden_neurons = list_check(hidden_neurons)
        neurons = [in_neurons] + hidden_neurons
        layers = []

        for in_n,out_n in zip(neurons,neurons[1:]):
            layers.extend([
                nn.Linear(in_n,out_n),
                nn.ReLU(),
                nn.Dropout(dropout),])
        layers.append(nn.Linear(neurons[-1],out_neurons))
        self.model = nn.Sequential(*layers)
        
    def forward(self,x) -> torch.Tensor:
        x = self.model(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self,in_neurons:int,
                 out_neurons:int,
                 hidden_neurons:int,
                 dropout:float,
                 use_layer_norm:bool):
        super().__init__()

        self.dense = nn.Sequential(
            nn.Linear(in_neurons,hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons,out_neurons),
            nn.Dropout(dropout),
        )

        self.skip = nn.Linear(in_neurons,out_neurons)
        self.layer_norm = nn.LayerNorm(out_neurons) if use_layer_norm else None

    def forward(self,x):
        x = self.dense(x) + self.skip(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)       
        return(x)
    
class ResNet(nn.Module):
    def __init__(self,
                 in_neurons:int,
                 hidden_neurons:int,
                 out_neurons:int,
                 no_blocks: int,
                 dropout:float = 0.,
                 use_layer_norm:bool = False,
                )->None:
        
        super().__init__()


        layers = [ResidualBlock(in_neurons,hidden_neurons,hidden_neurons,dropout,use_layer_norm)]
                
        for _ in range(no_blocks):
            layers.append(ResidualBlock(hidden_neurons,hidden_neurons,hidden_neurons,dropout,use_layer_norm))

        layers.append(ResidualBlock(hidden_neurons,hidden_neurons,out_neurons,dropout,use_layer_norm))
        self.model = nn.Sequential(*layers)

    def forward(self,x) -> torch.Tensor:
        x = self.model(x)     
        return x