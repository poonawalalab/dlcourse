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
    transforms.Normalize((0.1307,), (0.3081,))
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
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 7 * 7 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
model = Net()
print(model.parameters)

# 3. Loss and 4. Training

optimizer = optim.Adam(model.parameters())
for epoch in range(10):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            # Calculate test loss
            model.eval()
            with torch.no_grad():
                test_data, test_target = next(iter(test_loader))
                test_output = model(test_data)
                test_loss = F.nll_loss(test_output, test_target)
            model.train()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Test: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(),test_loss.item()))

# 5. Evaluate
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    for i in range(0,5):
        images, labels = next(iter(test_plot_loader))
        img, lbl = images[0].squeeze(), labels.item()
        plt.imshow(img, cmap="gray")
        plt.title(f"Random sample label: {lbl}")
        plt.axis("off")
        plt.show()

print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
    correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))


PATH = './mnist_model.pth'
torch.save(model.state_dict(), PATH)
