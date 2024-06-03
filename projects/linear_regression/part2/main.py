import training
import numpy as np
import getData
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(seed=1234)

# Load the data
train, test = getData.getDataSplits()

# Transform the data
x_train, y_train, x_test, y_test = getData.transformData(train, test)

regressor = training.LinearRegression()
regressor.fit(x_train, y_train)
y_hat = regressor.predict(x_test)

# plot target against predicted
y_test_sorted = np.sort(y_test)
idx = np.argsort(y_test)
y_hat_hat = y_hat[idx]

plt.title("predicted vs target")
plt.plot(y_test_sorted, "r.")
plt.plot(y_hat_hat, "b.")

# plot a line between the points
for i in range(len(y_test)):
    plt.plot([i, i], [y_test_sorted[i], y_hat_hat[i]], 'b-')

plt.show()
