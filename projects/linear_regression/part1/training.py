import pandas as pd
import numpy as np
import getData
import model
import matplotlib.pyplot as plt

learning_rate = 10e-8
max_epochs = 5_000_000
error_threshold = 10e-9
weight_decay = 5

# Set seed for reproducibility
np.random.seed(seed=1234)

train, test = getData.getDataSplits()

x_train, y_train, x_test, y_test = getData.transformData(train, test)

# add the bias term
w = np.array([1])
# add the weights for each feature randomly
w = np.append(w, np.random.rand(x_train.shape[1]))

# error matrix
error_matrix = np.array([])

w_hat, error_matrix = model.BatchGradientDescent(x_train, y_train, error_threshold, max_epochs, learning_rate, w, weight_decay, error_matrix)

# plot target against predicted
y_hat = np.dot(x_test, w_hat[1:]) + w_hat[0]

# order y_test
y_test = np.sort(y_test)
# order y_hat
y_hat = np.sort(y_hat)

# plot the target in red
plt.plot(y_test, 'ro')
# plot the predicted in blue
plt.plot(y_hat, 'bo')
plt.show()



