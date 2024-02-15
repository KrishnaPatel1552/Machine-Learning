import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class test:
    def __init__(self, batch_size=32, regularization=0, max_epochs=100, patience=3):
        """Linear Regression using Gradient Descent.

        Parameters:
        -----------
        batch_size: int
            The number of samples per batch.
        regularization: float
            The regularization parameter.
        max_epochs: int
            The maximum number of epochs.
        patience: int
            The number of epochs to wait before stopping if the validation loss
            does not improve.
        """
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        self.weights = None
        self.bias = None

    def fit(self, X, y, X_val=None, y_val=None):
        """Fit a linear model.

        Parameters:
        -----------
        X: numpy.ndarray
            The input features.
        y: numpy.ndarray
            The target values.
        X_val: numpy.ndarray, optional
            Validation set features.
        y_val: numpy.ndarray, optional
            Validation set target values.
        """
        # Initialize weights and bias
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        best_weights = self.weights
        best_bias = self.bias
        best_val_loss = float('inf')
        patience_count = 0

        # Training loop
        for epoch in range(self.max_epochs):
            # Shuffle the data (X, y)
            indices = np.arange(len(X))
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]

            # Mini-batch gradient descent
            for i in range(0, len(X), self.batch_size):
                X_batch = X[i:i + self.batch_size]
                y_batch = y[i:i + self.batch_size]

                # Compute predictions and gradients
                predictions = self.predict(X_batch)
                errors = predictions - y_batch

                # Update weights and bias using gradients and regularization
                gradient_weights = np.dot(X_batch.T, errors) / len(X_batch)
                gradient_bias = np.sum(errors) / len(X_batch)

                self.weights = (1 - self.regularization / len(X_batch)) * self.weights - gradient_weights
                self.bias = self.bias - gradient_bias

            # Calculate and monitor the training loss (MSE)
            training_loss = np.mean((self.predict(X) - y) ** 2)

            # Early stopping based on patience and validation loss
            if X_val is not None and y_val is not None:
                val_loss = np.mean((self.predict(X_val) - y_val) ** 2)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = self.weights
                    best_bias = self.bias
                    patience_count = 0
                else:
                    patience_count += 1

                if patience_count >= self.patience:
                    break

        # Set the best weights and bias
        self.weights = best_weights
        self.bias = best_bias

    def predict(self, X):
        """Predict using the linear model.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        """
        return np.dot(X, self.weights) + self.bias

    def score(self, X, y):
        """Evaluate the linear model using the mean squared error.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        y: numpy.ndarray
            The target data.
        """
        predictions = self.predict(X)
        mse = np.mean((predictions - y) ** 2)
        return mse

# Test the linear regression model using the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Using only the first feature for simplicity
X = X[:, :1]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Create and fit the linear regression model
linear_reg_model = test(batch_size=32, regularization=0.1, max_epochs=100, patience=3)
linear_reg_model.fit(X_train, y_train, X_val, y_val)

# Evaluate the model on the validation set
val_mse = linear_reg_model.score(X_val, y_val)
print(f"Validation Mean Squared Error: {val_mse}")
