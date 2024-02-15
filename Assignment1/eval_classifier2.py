from sklearn import datasets
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
from LogisticRegression import LogisticRegression

iris = datasets.load_iris()
X = iris.data[:, :2]  # Sepal length and width
y = (iris.target == 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=5, shuffle=False)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

plot_decision_regions(X_train, y_train, clf=log_reg, legend=2)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Logistic Regression - Sepal Features')
plt.savefig("Logistic Model 2.png")
plt.show()

accuracy = log_reg.score(X_test, y_test)
print("Test Accuracy (Sepal Features):", accuracy)
