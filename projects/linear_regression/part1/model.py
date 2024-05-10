import pandas as pd
import numpy as np
from typing import Tuple

# Set seed for reproducibility
np.random.seed(seed=1234)


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

def BatchGradientDescent(
    x_train: np.ndarray,
    y_train: np.ndarray,
    error_threshold: float,
    max_epochs: int,
    learning_rate: float,
    w: np.ndarray,
    weight_decay: float,
    error_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Batch Gradient Descent.

    Args:
        x_train: the train set
        y_train: the true values
        error_threshold: the error threshold
        max_epochs: the maximum number of epochs
        learning_rate: the learning rate
        w: the weights (bias + features weights)
        error_matrix: the error matrix

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
            print('we reached the max number of epochs')
            break

        y_hat = np.dot(x_train, w[1:]) + w[0]

        w[0] = w[0] - learning_rate / len(x_train) * np.sum(y_hat - y_train)

        w[1:] = w[1:] - learning_rate / len(x_train) * np.dot(
            (y_hat - y_train), x_train
        )

        # with weights decay
        w[1:] = w[1:]*(1 - learning_rate * weight_decay) - learning_rate/len(x_train) * np.dot((y_hat - y_train), x_train)

        error = MSE(y_train, y_hat)
        #error_matrix = np.append(error_matrix, error)

        delta_Error = abs(old_Error - error)
        old_Error = error

        if delta_Error < error_threshold:
            print("we reached the error threshold")
            print(f"Error: {error} Delta Error: {delta_Error} Epochs: {ephocs}")
        
        if(ephocs % 100000 == 0):
            print(f"Error: {error} Delta Error: {delta_Error} Epochs: {ephocs}")

    return w, error_matrix
