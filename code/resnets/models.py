import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResNetCIFAR(nn.Module):
    def __init__(self):
        super().__init__()
        resnet50 = models.resnet50(pretrained=True)
        resnet50.fc = torch.nn.Linear(resnet50.fc.in_features, 10)
        torch.nn.init.xavier_uniform_(resnet50.fc.weight)
        self.resnet50 = resnet50

    def forward(self,x):
        return self.resnet50(x)

class ResNet18CIFAR(nn.Module):
    def __init__(self):
        super().__init__()
        resnet18 = models.resnet18(pretrained=True)
        resnet18.fc = torch.nn.Linear(resnet18.fc.in_features, 10)
        torch.nn.init.xavier_uniform_(resnet18.fc.weight)
        self.resnet18 = resnet18

    def forward(self,x):
        return self.resnet18(x)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 100, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(100, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super().__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x
    

class ResNetInt(nn.Module):
    def __init__(self):
        super().__init__()

        self.de_layer3 = Interpolate(scale_factor=7, mode='bilinear')

        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, 10)
        nn.init.xavier_uniform_(self.resnet50.fc.weight)
        self.decoder = nn.Sequential(
            self.de_layer3,
            self.resnet50
        )
    
    def forward(self, x):
        return self.decoder(x)

