#Import bibliotek
import numpy as np
import matplotlib.pyplot as plt

#Przygotowanie danych
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=20, centers=2, random_state=123, cluster_std=1.5)

#Przygotowanie etykiet
y = np.where(y == 0, -1, 1)

#Definowanie klasy
class Perceptron:
  def __init__(self, learning_rate=0.01, n_iterations=10):
    self.learning_rate = learning_rate
    self.n_iterations = n_iterations
    self.weights = None
    self.bias = None

#Trening/metoda fit
  def fit(self, X, y):
    n_samples, n_features = X.shape
    self.weights = np.zeros(n_features)
    self.bias = 0

    for _ in range(self.n_iterations):
      for idx, x_i in enumerate(X):
        linear_output = np.dot(x_i, self.weights) + self.bias
        y_predicted = self.activation_function(linear_output)
        update = self.learning_rate * (y[idx] - y_predicted)
        self.weights += update * x_i
        self.bias += update

#Predykcja dla nowych danych
  def predict(self, X):
    linear_output = np.dot(X, self.weights) + self.bias
    y_predicted = self.activation_function(linear_output)
    return y_predicted

  def activation_function(self, x):
    return np.where(x >= 0, 1, 0)

#Uruchomienie instancji
perceptron = Perceptron(learning_rate=0.1, n_iterations=10)
perceptron.fit(X, y)

#Wizualizacja granicy decyzyjnej (matplotlib)
def plot_decision_boundary(X, y, classifier):
    x1 = np.arange(min(X[:, 0]) - 1, max(X[:, 0]) + 1, step=0.01)
    x2 = np.arange(min(X[:, 1]) - 1, max(X[:, 1]) + 1, step=0.01)
    xx, yy = np.meshgrid(x1, x2)
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='b')
    plt.show()


plot_decision_boundary(X, y, perceptron)
