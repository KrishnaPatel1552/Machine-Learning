import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions
from LogisticRegression import LogisticRegression

iris = datasets.load_iris()
X = iris.data[:, 2:4]  # petal length and width
y = (iris.target == 2).astype(int)  # Iris-Virginica

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=5, shuffle=False)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

plot_decision_regions(X_train, y_train, clf=log_reg)
plt.title('Logistic Regression - Petal Features')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Logistic Regression - petal Features')
plt.savefig("Logistic Model 1.png")
plt.show()

accuracy = log_reg.score(X_test, y_test)
print("Test Accuracy (Petal Features):", accuracy)
