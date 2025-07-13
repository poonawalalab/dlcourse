from typing import List
import torch
import torch.nn as nn
from utils import list_check

class mlpModel(nn.Module):
    def __init__(self,
                 in_neurons:int,
                 hidden_neurons:int|List[int],
                 out_neurons:int,
                 dropout:float = 0.0,
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