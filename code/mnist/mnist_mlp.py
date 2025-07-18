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
            nn.Linear(64, 10),  # 3 classes
        )

    def forward(self, x):
        return F.log_softmax(self.net(x), dim=1)

model = Net()
print(model.parameters)


## Evaluate for funs 
# x. Evaluate
correct = 0
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
        plt.imshow(img.view(28,28), cmap="gray")
        plt.title(f"Random sample true label: {lbl} predicted: {pred.item()}")
        plt.axis("off")
        plt.pause(1)
        print(f"Random sample true label: {lbl} predicted: {pred.item()}")

print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
    correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

# 3. Loss and 4. Training

# Setup for plotting
train_losses = []
test_losses = []
epochs_plot = []

plt.ion()
fig, ax = plt.subplots(figsize=(10, 6))

optimizer = optim.Adam(model.parameters())
for epoch in range(100):
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
            # Store losses for plotting
            current_epoch = epoch + batch_idx / len(train_loader)
            train_losses.append(loss.item())
            test_losses.append(test_loss.item())
            epochs_plot.append(current_epoch)

            
            # Update plot
            ax.clear()
            ax.plot(epochs_plot, train_losses, 'b-', label='Training Loss')
            ax.plot(epochs_plot, test_losses, 'r-', label='Test Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training and Test Loss vs Epoch')
            ax.legend()
            ax.grid(True)
            plt.pause(0.01)

            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Test: {:.6f}'.format(
            #     epoch, batch_idx * len(data), len(train_loader.dataset),
            #     100. * batch_idx / len(train_loader), loss.item(),test_loss.item()))

fig.savefig('test-train')
plt.show()

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
        plt.imshow(img.view(28,28), cmap="gray")
        plt.title(f"Random sample label: {lbl}")
        plt.axis("off")
        plt.show()

print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
    correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))


PATH = './mnist_model.pth'
torch.save(model.state_dict(), PATH)
