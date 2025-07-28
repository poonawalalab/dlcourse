import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from data import load_MNIST, eval_MNIST
from models_live import LeNet



_, _, test_dataset, test_loader, test_plot_loader = load_MNIST()

model = LeNet()


eval_MNIST(model,test_loader)

PATH = './mnist_model_cnn.pth'
model.load_state_dict(torch.load(PATH))

eval_MNIST(model,test_loader)
