from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

from LinearRegression import LinearRegression

iris = load_iris()
X, y = iris.data[:, [1]], iris.data[:, 3]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the trained model parameters
loaded_model = LinearRegression()
filepath = "model-four"
loaded_model.load(filepath + ".npz")  # Replace with the actual path
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)
# Evaluate the model on the test set
mse_test = loaded_model.score(X_test_scaled, y_test)

print("Mean Squared Error on Test Set:", mse_test)