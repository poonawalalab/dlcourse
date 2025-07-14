## Custom gradient descent to solve versicolor vs virginica classification using only four feature

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

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


# Load the dataset to get names
iris = load_iris()

# Access the data, only two species
iris_species_1 = 1
iris_species_2 = 2
X, y = load_two_species((iris_species_1, iris_species_2))

# Define the objective function and its gradient 
def L(w, b):
    e = y - 1/(1+np.exp(X @ w-b*np.ones(100)))
    return np.linalg.norm(e)/100

def grad_L(w, b):
    """Gradient of f"""
    e = y - 1/(1+np.exp(X @ w-b*np.ones(100)))
    act = 1/(1+np.exp(X @ w-b*np.ones(100)))
    grad_w = np.zeros(4)
    for i in range(0,4):
        grad_w[i] = np.sum(e * act * (1-act) * X[:,i])
    grad_b = -np.sum(e *act * (1-act))
    return np.concatenate([grad_w, np.array([grad_b])])*1/100

# Steepest Descent
def steepest_descent(start, alpha=0.1,tol=1e-6, max_iter=150000):
    theta = np.array(start, dtype=float)
    iterates = [theta.copy()]
    optimal_values=[L(theta[:4],theta[4])]
    for _ in range(max_iter):
        grad = grad_L(theta[:4], theta[4])
        
        # Newton step: x_new = x - H_inv * grad
        theta -= alpha*grad
        
        iterates.append(theta.copy())
        optimal_values.append(L(theta[:4],theta[4]))
        
        # Convergence check
        if np.linalg.norm(grad) < tol:
            break
    
    return theta, iterates, optimal_values


# Pick an initial point $\theta_0$
start_point = np.array([1.0,1.0,-5.0,0.0, -10.0])*0.0

# Run gradient descent 
optimum, iterates, optimal_values = steepest_descent(start_point)

# Extract iterate points for printing and plotting
iterates = np.array(iterates)
print(f"w = {iterates[-1,:4]}, b = {iterates[-1,4]}")

# Plotting: Loss
plt.figure(figsize=(8, 6))
plt.plot(range(0,len(optimal_values)),optimal_values,  'o-', color="blue")
plt.xlabel("iterates (k)")
plt.ylabel(f"Loss")
plt.show()


# Show Confusion Matrix
w = iterates[-1,:4] 
b = iterates[-1,4]
y_pred = 1/(1+np.exp(X @ w -b*np.ones(100)))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y,np.round(y_pred))
print("Confusion Matrix:\n",cm)
