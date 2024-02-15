from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

iris = load_iris()
model = LinearRegression(regularization=0.1)
model.fit(iris.data[:, 1:3], iris.data[:, 0])
non_regularized_model = np.load('model3.npz')
non_reg_weights = non_regularized_model['weights']
non_reg_bias = non_regularized_model['bias']
model_name = "model 3 regularized"
plt.plot(range(len(model.loss_history)), model.loss_history, label=model_name)
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.savefig(model_name + ".png")
plt.show()

print("\nNon-regularized Model Weights:")
print("Weights:", non_reg_weights)
print("Bias:", non_reg_bias)

print("\nRegularized Model Weights:")
print("Weights:", model.weights)
print("Bias:", model.bias)

print("\nDifference between Model Weights and bias:")
print("Weights:", np.sum(np.abs(non_reg_weights - model.weights)))
print("Bias:", np.abs(non_reg_bias - model.bias))
