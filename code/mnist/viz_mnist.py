import torch
from torchvision.datasets import MNIST
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# 1. Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)


random_array = np.random.randint(0, 59999, size=10)
print(random_array)

for i in range(10):
    image, label = train_dataset[random_array[i]]
    if i == 0:
        print("tensor shape of image (not plottable): ",image.shape)
        print("tensor shape of image[0] (plottable): ",image[0].shape)
    plt.imshow(image[0], cmap="gray")
    plt.title(f"Random sample {i} label: {label}")
    plt.pause(1)
