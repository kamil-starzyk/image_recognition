import numpy as np

def unit_step_func(x):
  return np.where(x > 0, 1, 0) #where x > 0 return 1 else return 0

class Perceptron:
  def __init__(self, learning_rate=0.1, n_iters=1000):
    self.lr = learning_rate
    self.n_iters = n_iters
    self.activation_function = unit_step_func
    self.weights = None
    self.bias = None
  
  def fit(self, X, y):
    #n_samples is the number of rows in the matrix, often representing the number of samples or data points.
    #n_features is the number of columns in the matrix, representing the number of features or variables for each sample.
    n_samples, n_features = X.shape #X is numpy N-dimensional array
    print(f'samples {n_samples}, features {n_features}')


    #init parameters
    #self.weights = np.zeros(n_features)
    self.weights = np.random.rand(n_features)
    self.bias = 0

    y = np.where(y > 0, 1, 0) #where y > 0 return 1 else return 0

    #learning weights
    for _ in range(self.n_iters):
      for idx, x_i in enumerate(X): #idx - index, x_i - sample
        linear_output = np.dot(x_i, self.weights) + self.bias
        y_predicted = self.activation_function(linear_output)

        #perceptron update rule
        update = self.lr * (y[idx] - y_predicted)
        self.weights += update * x_i
        self.bias += update
    

  def predict(self, X):
    linear_output = np.dot(X, self.weights) + self.bias
    y_predicted = self.activation_function(linear_output)
    return y_predicted


#testing
if __name__ == "__main__":
  import matplotlib.pyplot as plt
  from sklearn.model_selection import train_test_split
  from sklearn import datasets

  def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy
  
  #zwiększając cluster_std do wartości ponad 2 wygenerujemy obsdzary nachodzące na siebie co utrudni wyznaczenie jednoznacznej linii  decyzyjnej
  #co poskutkuje zmniejszeniem accuracy
  X, y = datasets.make_blobs(n_samples=150, n_features=2, centers=2, cluster_std=2.05, random_state=2)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
  print(len(X_train))
  print(len(X_test))
  print(X)
  print(y)

  p = Perceptron(learning_rate=0.01, n_iters=1000)
  p.fit(X_train, y_train)
  predictions = p.predict(X_test)

  print("Accuracy: ", accuracy(y_test, predictions))

  fig = plt.figure()
  ax = fig.add_subplot(1,1,1)

  plt.scatter(X_train[:, 0], X_train[:, 1], marker="x", c=y_train) # -- na wykresie są narysowane punkty treningowe
  plt.scatter(X_test[:, 0], X_test[:, 1], marker="o", c=y_test)

  x0_1 = np.amin(X_train[:, 0])
  x0_2 = np.amax(X_train[:, 0])

  x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
  x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]

  ax.plot([x0_1, x0_2], [x1_1, x1_2], "k") # -- oraz linia decyzyjna

  ymin = np.amin(X_train[:, 1])
  ymax = np.amax(X_train[:, 1])

  ax.set_ylim([ymin - 3, ymax + 3])

  plt.show()
 
