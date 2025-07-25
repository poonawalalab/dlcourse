import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Sigmoid(),
            nn.Flatten(),
            nn.Linear(400,120),
            nn.Sigmoid(),
            nn.Linear(120,84),
            nn.Sigmoid(),
            nn.Linear(84,10),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)
