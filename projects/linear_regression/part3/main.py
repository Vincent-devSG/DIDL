import torch
import torch.nn as nn
from getData import getDataSplits, transformData
from model import LinearRegression, train_model
import  numpy as np
import matplotlib.pyplot as plt



def main():
    # Load and transform data
    train, test = getDataSplits()
    x_train, y_train, x_test, y_test = transformData(train, test)

    # Set input and output dimensions
    input_dim = x_train.shape[1]
    output_dim = 1  # Single output for regression

    # Initialize the model
    model = LinearRegression(input_dim, output_dim)

    # Train the model
    train_model(model, x_train, y_train, x_test, y_test, epochs=50, lr=10e-2, weight_decay=1e-2)

    print("Model training complete.")

    y_hat = model(x_test).detach().numpy()

    y_test_sorted = np.sort(y_test.squeeze())
    idx = np.argsort(y_test.squeeze())
    y_hat_hat = y_hat[idx].squeeze()

    plt.title("predicted vs target")
    plt.plot(y_test_sorted, 'r.')
    plt.plot(y_hat_hat, 'b.')

    for i in range(len(y_test_sorted)):
        plt.plot([i, i], [y_test_sorted[i], y_hat_hat[i]], 'b-')

    plt.show()

if __name__ == "__main__":
    main()
