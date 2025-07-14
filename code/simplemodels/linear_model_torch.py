## Train a linear model using pytorch

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 1. Create data
## First as numpy array
x = np.linspace(-1,1,100) 
y_noise_free = 4*x + 5
y = y_noise_free+0.5*np.random.randn(100)


## Create tensors, as 2D arrays using unsqueeze 
X_train = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
y_train = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

print(X_train.shape) ## two dimensions now
print(y_train.shape)


# 2. Define Linear model
class Lin(nn.Module):
    def __init__(self):
        super().__init__()
        self.net =  nn.Linear(1, 1)

    def forward(self, x):
        return self.net(x)
model = Lin()

# 3. Define loss 
criterion = nn.MSELoss() # Mean Squared Error

print("Start training: ")
# 4. Training loop
for epoch in range(5000):
    model.train()       # Tell pytorch that model computations are for training now
    model.zero_grad()   # clear accumulated grads
    outputs = model(X_train) # predict values ... 
    loss = criterion(outputs, y_train) # ... so we can compute loss 
    loss.backward()     # Automatically get gradients of loss wrt model as it is right now
    with torch.no_grad(): ## Manually implementing gradient descent. Don't want the update to be tracked by autograd
        for param in model.parameters():
            param -= 0.005 * param.grad  # θ ← θ − α∇θ L

    if (epoch + 1) % 500 == 0: ## Show us the loss as training proceeds
        print(f"Epoch {epoch+1}/5000 - Loss: {loss.item():.4f}")

# 5. Evaluation
model.eval()    # tell pytorch that we will use the model in evaluation mode
with torch.no_grad():
    outputs = model(X_train).numpy() # convert predictions to numpy array to work with matplotlib
    plt.scatter(x,y,label="noise data")
    plt.scatter(x,y_noise_free,label="true (noise-free)")
    plt.scatter(x,outputs[:,0],label="predicted")
    plt.legend()

