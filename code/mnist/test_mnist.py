## See `./code/mnist/test_mnist.py`
import torch
from torchvision.datasets import MNIST
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 1. Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
test_plot_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)



# 2. Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.Sigmoid(),
            nn.Linear(512, 256),
            nn.Sigmoid(),
            nn.Linear(256, 128),
            nn.Sigmoid(),
            nn.Linear(128, 64),
            nn.Sigmoid(),
            nn.Linear(64, 10),  # 10 classes
        )

    def forward(self, x):
        return F.log_softmax(self.net(x), dim=1)

model = Net()

PATH = './mnist_model.pth'
model.load_state_dict(torch.load(PATH))

# x. Evaluate
correct = 0

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(16, 8))
axes = axes.flatten()

with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    for i in range(0,10):
        images, labels = next(iter(test_plot_loader))
        output = model(images)
        pred = output.argmax(dim=1, keepdim=True)
        img, lbl = images[0].squeeze(), labels.item()
        axes[i].imshow(img.view(28,28), cmap="gray")
        axes[i].set_title(f"true: {lbl} predicted: {pred.item()}")
    plt.show()

print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
    correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
