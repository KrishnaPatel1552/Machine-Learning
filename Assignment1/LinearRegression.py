import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


class LinearRegression:
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
        self.loss_history = []
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        self.weights = None
        self.bias = None

    def fit(self, X, y, batch_size=32, regularization=0.0, max_epochs=100, patience=3, learning_rate=0.001):
        """Fit a linear model.

        Parameters:
        -----------
        X: numpy.ndarray
            The input data.
        y: numpy.ndarray
            The target values.
        batch_size: int
            The number of samples per batch.
        regularization: float
            The regularization parameter.
        max_epochs: int
            The maximum number of epochs.
        patience: int
            The number of epochs to wait before stopping if the validation loss
            does not improve.
        learning_rate: float
            The learning rate for gradient descent.
        """
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience

        self.weights = np.random.randn(X.shape[1])
        self.bias = np.random.randn()

        X_training_set, X_validation_set, y_training_set, y_validation_set = train_test_split(X, y, test_size=0.1,
                                                                                              random_state=5,
                                                                                              shuffle=False)

        weights = self.weights.copy()
        bias = self.bias
        val_loss_best = float('inf')
        step = 0

        for epoch in range(self.max_epochs):
            for i in range(0, len(X_training_set), self.batch_size):
                X_batch = X_training_set[i:i + self.batch_size]
                y_batch = y_training_set[i:i + self.batch_size]
                predictions = np.dot(X_batch, self.weights) + self.bias

                error = predictions - y_batch
                gradient_weights = (2 / len(X_batch)) * np.dot(X_batch.T, error)
                gradient_bias = (2 / len(X_batch)) * np.sum(error)
                gradient_weights += 2 * self.regularization * self.weights
                self.weights -= learning_rate * gradient_weights
                self.bias -= learning_rate * gradient_bias

            val_predictions = self.predict(X_validation_set)
            val_loss = np.mean((val_predictions - y_validation_set) ** 2)
            self.loss_history.append(val_loss)

            if val_loss < val_loss_best:
                val_loss_best = val_loss
                weights = self.weights.copy()
                bias = self.bias
                step = 0
            else:
                step += 1
                if step >= patience:
                    break

        self.weights = weights
        self.bias = bias

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

    def save(self, file_path):
        """Save the model parameters to a file.

        Parameters:
        -----------
        file_path: str
            The file path to save the model parameters.
        """
        np.savez(file_path, weights=self.weights, bias=self.bias)

    def load(self, file_path):
        """Load the model parameters from a file.

        Parameters:
        -----------
        file_path: str
            The file path to load the model parameters.
        """
        data = np.load(file_path)
        self.weights = data['weights']
        self.bias = data['bias']


def plot_loss(model, model_name):
    plt.plot(model.loss_history, label=model_name)
    plt.xlabel('Step No.')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(model_name + ".png")
    plt.show()


def main():
    # Load Iris dataset
    iris = load_iris()

    model1 = LinearRegression()
    model2 = LinearRegression()
    model3 = LinearRegression()
    model4 = LinearRegression()

    model1.fit(iris.data[:, 2:3], iris.data[:, 1])
    model2.fit(iris.data[:, :2], iris.data[:, 1])
    model3.fit(iris.data[:, 1:3], iris.data[:, 0])
    model4.fit(iris.data[:, 0:2], iris.data[:, 3])

    model1.save('model1.npz')
    model2.save('model2.npz')
    model3.save('model3.npz')
    model4.save('model4.npz')

    plot_loss(model1, 'Model 1 - petal length vs sepal width')
    plot_loss(model2, 'Model 2 - sepal length and width vs petal length')
    plot_loss(model3, 'Model 3 - sepal width and petal length vs sepal length')
    plot_loss(model4, 'Model 4 - sepal length and width vs petal width')


if __name__ == '__main__':
    main()
