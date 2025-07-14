## Custom code to train a linear model using gradient descent
import numpy as np
import matplotlib.pyplot as plt

### Generate data
x = np.linspace(-1,1,100) 
y = 4*x + 5+0.5*np.random.randn(100)


# Define the loss function and its gradient 
def L(a, b):
    e = y - a*x-b*np.ones(100)
    return np.linalg.norm(e)/100

def grad_L(a, b):
    """Gradient of f"""
    e = y - a*x-b*np.ones(100)
    grad_x = -np.sum(e * x)
    grad_y = -np.sum(e)
    return np.array([grad_x, grad_y])

# Steepest Descent implementation
def steepest_descent(start, alpha=0.005,tol=1e-6, max_iter=5000):
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
start_point = [3.0, 1.0]

# Run gradient descent 
optimum, iterates, optimal_values = steepest_descent(start_point)

# Extract iterate points for printing and plotting
iterates = np.array(iterates)
x_iterates, y_iterates = iterates[:, 0], iterates[:, 1]
print(f"a = {x_iterates[-1]}, b = {y_iterates[-1]}")

# Plotting: Iterates
plt.figure(figsize=(8, 6))
plt.plot(4, 5, 'x', color="blue", markersize=10, label="True Optimum (4, 5)")
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
