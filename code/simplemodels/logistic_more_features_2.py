#print(iris.feature_names)
#print(iris.target_names)

# Access the data, only two species
iris_species_1 = 1
iris_species_2 = 2
X, y = load_two_species((iris_species_1, iris_species_2))

# Define the objective function and its gradient 
def f(w, b):
    e = y - 1/(1+np.exp(X @ w-b*np.ones(100)))
    return np.linalg.norm(e)/100

def grad_f(w, b):
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
    optimal_values=[f(theta[:4],theta[4])]
    for _ in range(max_iter):
        grad = grad_f(theta[:4], theta[4])
        
        # Newton step: x_new = x - H_inv * grad
        theta -= alpha*grad
        
        iterates.append(theta.copy())
        optimal_values.append(f(theta[:4],theta[4]))
        
        # Convergence check
        if np.linalg.norm(grad) < tol:
            break
    
    return theta, iterates, optimal_values
# Run Gradient Descent
start_point = np.array([1.0,1.0,-5.0,0.0, -10.0])*0.0
# predicted first grad: [0.04075 0.01275 0.08075 0.04375]
optimum, iterates, optimal_values = steepest_descent(start_point)

# Extract iterate points for plotting
iterates = np.array(iterates)
print(f"w = {iterates[-1,:4]}, b = {iterates[-1,4]}")

plt.figure(figsize=(8, 6))
plt.plot(range(0,len(optimal_values)),optimal_values,  'o-', color="blue")
plt.xlabel("iterates (k)")
plt.ylabel(f"Loss")
plt.show()


w = iterates[-1,:4] 
b = iterates[-1,4]
#w=np.array([0.0,0.0,-5.4,0.0])
#b=-14
y_pred = 1/(1+np.exp(X @ w -b*np.ones(100)))
# sum = 0
# print(" pred   truth")
# for i in range(0,100):
# 	print(f"{y_pred[i]: .3f} {y[i]: .0f}")
# 	if abs(y_pred[i] - y[i])<0.1:
# 		sum+=1
#print(f"sum: {sum}")
cm = confusion_matrix(y,np.round(y_pred))
print("Confusion Matrix:\n",cm)
