import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the dataset to get names
iris = load_iris()


Xplot = iris.data ## '2'	: petal length (cm) 

for i in range(0,4):
    for j in range(i+1,4):
        plt.figure(figsize=(8, 6))
        plt.scatter(Xplot[:50,i],Xplot[:50,j],label=f"{iris.target_names[0]}")
        plt.scatter(Xplot[50:100,i],Xplot[50:100,j],label=f"{iris.target_names[1]}")
        plt.scatter(Xplot[100:,i],Xplot[100:,j],label=f"{iris.target_names[2]}")
        plt.xlabel(iris.feature_names[i])
        plt.ylabel(iris.feature_names[j])
        plt.legend()
        plt.show()
