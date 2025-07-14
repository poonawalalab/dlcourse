## Custom gradient descent to solve setosa vs versicolor classification using only one feature
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

# Access part of the data
iris_single_feature = 2 
iris_species_1 = 0 ## setosa
iris_species_2 = 1 ## versicolor 
X, y = load_two_species((iris_species_1, iris_species_2))
x = X[:,iris_single_feature] ## pick one column of data corresponding to a feature




# Define the loss function and its gradient
def L(a, b):
    e = y - 1/(1+np.exp(a*x-b*np.ones(100)))
    return np.linalg.norm(e)/100

def grad_L(a, b):
    """Gradient of f"""
    e = y - 1/(1+np.exp(a*x-b*np.ones(100)))
    act = 1/(1+np.exp(a*x-b*np.ones(100)))
    grad_x = np.sum(e * act * (1-act) * x)
    grad_y = -np.sum(e * act * (1-act))
    return np.array([grad_x, grad_y])*1/100

# Steepest Descent
def steepest_descent(start, alpha=0.1,tol=1e-6, max_iter=50000):
    theta = np.array(start, dtype=float)
    iterates = [theta.copy()]
    optimal_values=[L(theta[0],theta[1])]
    for _ in range(max_iter):
        grad = grad_L(theta[0], theta[1])
        
        # Descent step: theta_(k+1) = theta_k - alpha * grad
        theta -= alpha*grad
        
        iterates.append(theta.copy())
        optimal_values.append(L(theta[0],theta[1]))
        
        # Convergence check
        if np.linalg.norm(grad) < tol:
            break
    
    return theta, iterates, optimal_values

# Pick an initial point $\theta_0$
start_point = [-3.0, 1.0]
# Plotting

# Run gradient descent 
optimum, iterates, optimal_values = steepest_descent(start_point)
# Extract iterate points for printing and  plotting
iterates = np.array(iterates)
x_iterates, y_iterates = iterates[:, 0], iterates[:, 1]
print(f"a = {x_iterates[-1]}, b = {y_iterates[-1]}")


# Plotting: Iterates
plt.figure(figsize=(8, 6))
plt.plot(x_iterates, y_iterates, 'o-', color="blue", label="SD Iterates")
# Labels and legend
plt.xlabel("a")
plt.ylabel("b")
plt.title("Steepest Descent")
plt.legend()
plt.grid(True)
plt.show()

# Plotting: Loss
plt.figure(figsize=(8, 6))
plt.plot(range(0,len(optimal_values)),optimal_values,  'o-', color="blue")
plt.xlabel("iterates (k)")
plt.ylabel(f"Loss")
plt.show()
