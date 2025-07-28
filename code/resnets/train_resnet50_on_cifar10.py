## Trains a model-mod where the final layer output size is reduced for CIFAR10, and CIFAR10 images are resized to 224x224. 
## We only train the fc layer
## Based on https://pytorch.org/hub/nvidia_deeplearningexamples_model/
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
from models import ResNet18CIFAR as ResNetCIFAR

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

    print('Accuracy before training')
    test_cifar(model,testloader,device,classes)
    show_cifar(model,testloader,device,classes)



    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    print('Start training') 
    model.train()
    for epoch in range(30):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            # inputs, labels = data
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    PATH = './cifar_model_v4.pth'
    torch.save(model.state_dict(), PATH)



    print('Accuracy after training')
    test_cifar(model,testloader,device,classes)
    show_cifar(model,testloader,device,classes)
