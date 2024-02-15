from sklearn import datasets
from sklearn.model_selection import train_test_split
from LogisticRegression import LogisticRegression

iris = datasets.load_iris()
X = iris.data  # All features
y = (iris.target == 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=5, shuffle=False)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

accuracy = log_reg.score(X_test, y_test)
print("Test Accuracy (All Features):", accuracy)
