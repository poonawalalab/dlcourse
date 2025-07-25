import torch
from torchvision.datasets import MNIST
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from data import load_MNIST, eval_MNIST
from models import LeNet



train_dataset, train_loader, test_dataset, test_loader, test_plot_loader = load_MNIST()

model = LeNet()


eval_MNIST(model,test_loader)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
for epoch in range(10):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            # Calculate test loss
            model.eval()
            with torch.no_grad():
                test_data, test_target = next(iter(test_loader))
                test_output = model(test_data)
                test_loss = criterion(test_output, test_target)
            model.train()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Test: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(),test_loss.item()))



eval_MNIST(model,test_loader)
PATH = './mnist_model_cnn.pth'
torch.save(model.state_dict(), PATH)
