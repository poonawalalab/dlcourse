## This code classifies versicolor vs virginica using PyTorch

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np
from utils import load_two_species

# 1. Load and preprocess data
iris = load_iris()

iris_species_1 = 1
iris_species_2 = 2

X, y = load_two_species((iris_species_1, iris_species_2))

X_train = torch.tensor(X, dtype=torch.float32)
y_train = torch.tensor(y, dtype=torch.float32).unsqueeze(1)



# 2. Define MLP model: Perceptron = linear + sigmoid
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)
model = MLP()

with torch.no_grad(): ## Manual initialization to zero
    model.net[0].weight[0,0] = 0.
    model.net[0].weight[0,1] = 0.
    model.net[0].weight[0,2] = 0.0
    model.net[0].weight[0,3] = 0.
    model.net[0].bias[0] = 0.0

state = model.state_dict()
print("Initial w: ",state['net.0.weight'], "b:", state['net.0.bias'])

# 3. Define loss 
criterion = nn.MSELoss()

print("Start training: ")
# 4. Training loop
for epoch in range(150000):
    model.train()
    model.zero_grad()  # clear accumulated grads
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    with torch.no_grad(): ## Manually implementing gradient descent. Don't want the update to be tracked by autograd
        for param in model.parameters():
            param -= 0.1 * param.grad  # θ ← θ − α∇θ L
    # --------------------------------------

    if (epoch + 1) % 15000 == 1:
        print(f"Epoch {epoch+1}/150000 - Loss: {loss.item():.4f}")

# 5. Evaluation
model.eval()
with torch.no_grad():
    outputs = model(X_train)
    print("\nClassification Report:\n")
    print(classification_report(y_train.numpy().round(), outputs.numpy().round(), target_names=iris.target_names[1:]))

