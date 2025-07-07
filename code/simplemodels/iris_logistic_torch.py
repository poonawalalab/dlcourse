import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np

# 1. Load and preprocess data
iris = load_iris()

def load_two_species(species_pair=(0, 1), *, remap=True):
    if len(species_pair) != 2:
        raise ValueError("species_pair must contain exactly two integers (e.g. (0, 2)).")
    if not all(label in {0, 1, 2} for label in species_pair):
        raise ValueError("Labels must be chosen from 0, 1, 2.")

    iris = load_iris()
    X_all, y_all = iris.data, iris.target

    # Boolean mask: keep rows whose label is in species_pair
    mask = np.isin(y_all, species_pair)
    X, y = X_all[mask], y_all[mask]

    if remap:
        # Map the first chosen label → 0, the second → 1
        label_map = {species_pair[0]: 0, species_pair[1]: 1}
        y = np.vectorize(label_map.get)(y)

    return X, y
iris_species_1 = 1
iris_species_2 = 2

X, y = load_two_species((iris_species_1, iris_species_2))

# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# print("X:", X)
# print("pred", (np.sum(X[50:,:],axis=0)-np.sum(X[:50],axis=0))*1/800)

X_train = torch.tensor(X, dtype=torch.float32)
y_train = torch.tensor(y, dtype=torch.float32).unsqueeze(1)



# 2. Define MLP model
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

