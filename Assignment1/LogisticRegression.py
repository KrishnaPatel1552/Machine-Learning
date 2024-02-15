import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        self.weights = np.zeros(X_bias.shape[1])

        # Gradient Descent
        for _ in range(self.n_iterations):
            predictions = self._sigmoid(np.dot(X_bias, self.weights))
            errors = y - predictions
            gradient = np.dot(X_bias.T, errors) / len(y)
            self.weights += self.learning_rate * gradient

    def predict(self, X):
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        predictions = self._sigmoid(np.dot(X_bias, self.weights))
        return (predictions >= 0.5).astype(int)

    def score(self, X, y):
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy

    def save_weights(self, filename):
        np.savez(filename, weights=self.weights)

    def load_weights(self, filename):
        data = np.load(filename)
        self.weights = data['weights']

