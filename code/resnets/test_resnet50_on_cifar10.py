# tests resnet50 on CIFAR10 without using utils
## Based on https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
## Modified by Hasan Poonawala 

import torch
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import numpy as np
import json
import requests
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os
from data import load_cifar, test_cifar, show_cifar
from utils import imshow
from models import ResNetCIFAR, ResNet18CIFAR

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


if __name__ == "__main__":

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256mb"
    torch.cuda.empty_cache()
    #1. Load Data

    transform = transforms.Compose(
        [torchvision.transforms.Resize((224,224)), 
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    classes, trainset, trainloader, testset, testloader = load_cifar(transform,batch_size=16)


    device = torch.device("mps") if torch.mps.is_available() else torch.device("cpu")
    print(f'Using {device} for inference')

    # PATH = 'cifar_model_v3.pth'
    # model.load_state_dict(torch.load(PATH))
    model = ResNetCIFAR()
    model.to(device)

    test_cifar(model,testloader,device,classes)
