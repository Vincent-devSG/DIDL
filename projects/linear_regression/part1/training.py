import pandas as pd
import numpy as np
import getData
import matplotlib.pyplot as plt


learning_rate = 10e-6
max_epochs = 100_000
error_threshold = 10e-6

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


def MSE(y, y_hat):
    """
    Compute the Mean Squared Error.

    Args:
        y: the true values
        y_hat: the predicted values

    Returns:
        The Mean Squared Error
    """
    return np.mean((y - y_hat) ** 2)


def BatchGradientDescent(error_matrix=error_matrix):
    """
    Compute the Batch Gradient Descent.

    Args:
        x: the features
        y: the true values
        w: the weights
        learning_rate: the learning rate

    Returns:
        None
    """
    # Init parameters
    delta_Error = 10e3
    old_Error = 10e3
    ephocs = 0

    while delta_Error > error_threshold:

        # if we reach the max number of epochs, we stop
        ephocs += 1
        if ephocs > max_epochs:
            break

        y_hat = np.dot(x_train, w[1:]) + w[0]

        w[0] = w[0] - learning_rate / len(x_train) * np.sum(y_hat - y_train)

        w[1:] = w[1:] - learning_rate / len(x_train) * np.dot(
            (y_hat - y_train), x_train
        )

        error = MSE(y_train, y_hat)
        error_matrix = np.append(error_matrix, error)

        delta_Error = abs(old_Error - error)
        old_Error = error

        if delta_Error < error_threshold:
            print("we reached the error threshold")
            print(f"Error: {error} Delta Error: {delta_Error} Epochs: {ephocs}")
            print(y_hat[:5])
            print(y_train[:5])

    return w, error_matrix


w_hat, error_matrix = BatchGradientDescent()

too_big = [0, 1, 2]
error_matrix = np.delete(error_matrix, too_big)
plt.plot(error_matrix)
plt.show()
