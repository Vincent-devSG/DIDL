import numpy as np
import json
from model import BatchGradientDescent


class LinearRegression:
    """Class for Linear Regression model.
    
    Attributes:
        learning_rate (float): The learning rate for gradient descent.
        max_epochs (int): The maximum number of epochs for training.
        weight_decay (float): The weight decay (L2 penalty) parameter.
    """

    def __init__(
        self,
        learning_rate: float = None,
        max_epochs: int = None,
        weight_decay: float = None,
        error_threshold: float = None,
    ) -> None:
        """Linear Regression model constructor.

        It initializes the Linear Regression model with the given parameters.
        If no parameters are given, it reads the parameters from a JSON file.
        If parameters are given, it uses them.
        If partial parameters are given, it reads the rest from the JSON file.

        Parameters:
        - learning_rate (float): The learning rate for gradient descent. Default is None.
        - max_epochs (int): The maximum number of epochs for training. Default is None.
        - weight_decay (float): The weight decay (L2 penalty) parameter. Default is None.
        """
        with open("setup.json", "r") as f:
            setups = json.load(f)
            self.learning_rate = setups["learning_rate"]
            self.max_epochs = setups["max_epochs"]
            self.weight_decay = setups["weight_decay"]
            self.error_threshold = setups["error_threshold"]

        if learning_rate is not None:
            self.learning_rate = learning_rate
        if max_epochs is not None:
            self.max_epochs = max_epochs
        if weight_decay is not None:
            self.weight_decay = weight_decay
        
        # The weights array, subsuming the bias term
        self.weigths = None

        # The error matrix
        self.error_matrix = None
        
    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the Linear Regression model using the given train set.

        It trains the Linear Regression model using the given train set.

        Parameters:
        - x_train (np.ndarray): The train set.
        - y_train (np.ndarray): The true values.
        """
        # add the bias term
        self.weigths = np.array([1])
        # add the weights for each feature randomly
        self.weigths = np.append(self.weigths, np.random.rand(x_train.shape[1]))
        # error matrix
        self.error_matrix = np.array([])

        self.weigths, self.error_matrix = BatchGradientDescent(
            x_train, y_train, self.error_threshold, self.max_epochs, self.learning_rate, self.weigths, self.weight_decay, self.error_matrix
        )

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        """Predict the target values using the given test set.

        It predicts the target values using the given test set.

        Parameters:
        - x_test (np.ndarray): The test set.

        Returns:
        - np.ndarray: The predicted target values.
        """
        y_hat = np.dot(x_test, self.weigths[1:]) + self.weigths[0]

        # Negative values are not possible for the target also value should be between 0 and 1
        # so we clip the values
        y_hat = np.clip(y_hat, 0, 1)

        return y_hat

