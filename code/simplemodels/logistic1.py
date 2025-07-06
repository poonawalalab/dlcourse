# Define the objective function and its gradient and Hessian
def f(a, b):
    e = y - 1/(1+np.exp(a*x-b*np.ones(100)))
    return np.linalg.norm(e)/100

def grad_f(a, b):
    """Gradient of f"""
    e = y - 1/(1+np.exp(a*x-b*np.ones(100)))
    act = 1/(1+np.exp(a*x-b*np.ones(100)))
    grad_x = -np.sum(e * act * (1-act) * x)
    grad_y = -np.sum(e)
    return np.array([grad_x, grad_y])

# Steepest Descent
def steepest_descent(start, alpha=0.005,tol=1e-6, max_iter=5000):
    theta = np.array(start, dtype=float)
    iterates = [theta.copy()]
    optimal_values=[f(theta[0],theta[1])]
    for _ in range(max_iter):
        grad = grad_f(theta[0], theta[1])
        
        # Newton step: x_new = x - H_inv * grad
        theta -= alpha*grad
        
        iterates.append(theta.copy())
        optimal_values.append(f(theta[0],theta[1]))
        
        # Convergence check
        if np.linalg.norm(grad) < tol:
            break
    
    return theta, iterates, optimal_values
# Run Newton's method
start_point = [-3.0, 1.0]
# Plotting

plt.figure(figsize=(8, 6))
# Contour plot of the objective function

# Plot the iterates


optimum, iterates, optimal_values = steepest_descent(start_point)
# Extract iterate points for plotting
iterates = np.array(iterates)
x_iterates, y_iterates = iterates[:, 0], iterates[:, 1]
plt.plot(x_iterates, y_iterates, 'o-', color="blue", label="SD Iterates")

# Labels and legend
plt.xlabel("a")
plt.ylabel("b")
plt.title("Steepest Descent")
plt.legend()
plt.grid(True)
plt.show()

print(f"a = {x_iterates[-1]}, b = {y_iterates[-1]}")



```


```{python}
plt.figure(figsize=(8, 6))
plt.plot(range(0,len(optimal_values)),optimal_values,  'o-', color="blue")
plt.xlabel("iterates (k)")
plt.ylabel(f"Loss")
plt.show()
