## Custom gradient descent to solve setosa vs versicolor classification using all four feature
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


iris = load_iris() 
# Here, we manually pick out first two classes, given 50 measurements from each class.
x = iris.data[:100,2] ## '2'	: petal length (cm) 
X = iris.data[:100,:] ## '2'	: petal length (cm) 
y = iris.target[:100] ## ':100' : for 'setosa' and 'versicolor' only


# Define the loss function and its gradient 
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
    grad_b = -np.sum(e * act * (1-act))
    return np.concatenate([grad_w, np.array([grad_b])])*1/100

# Steepest Descent
def steepest_descent(start, alpha=0.1,tol=1e-6, max_iter=15000):
    theta = np.array(start, dtype=float)
    iterates = [theta.copy()]
    optimal_values=[L(theta[:4],theta[4])]
    for _ in range(max_iter):
        grad = grad_L(theta[:4], theta[4])
        
        # Descent step: theta_(k+1) = theta_k - alpha * grad
        theta -= alpha*grad
        
        iterates.append(theta.copy())
        optimal_values.append(L(theta[:4],theta[4]))
        
        # Convergence check
        if np.linalg.norm(grad) < tol:
            break
    
    return theta, iterates, optimal_values

# Pick an initial point $\theta_0$
start_point = np.zeros(5)

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
