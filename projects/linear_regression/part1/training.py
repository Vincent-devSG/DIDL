import pandas as pd
import numpy as np
import getData
import model
import matplotlib.pyplot as plt
import timeit
import seaborn as sns


# start the timer
start = timeit.default_timer()

learning_rate = 10e-6
max_epochs = 10_000_000
error_threshold = 10e-10
weight_decay = 10e-2

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

# stop the timer
stop = timeit.default_timer()
print('Time: ', stop - start)


# plot target against predicted
y_hat = np.dot(x_test, w[1:]) + w[0]

# Negative values are not possible for the target also value should be between 0 and 1 
# so we clip the values
y_hat = np.clip(y_hat, 0, 1)

y_test_sorted = np.sort(y_test)
idx = np.argsort(y_test)
y_hat_hat = y_hat[idx]

# calculate the R-squared
r_squared = 1 - np.sum((y_test - y_hat) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
print(f'R-squared: {r_squared}')

# plot the target in red
plt.plot(y_test_sorted, 'r.')
# plot the predicted in blue
plt.plot(y_hat_hat, 'b.')

# plot a line between the points
for i in range(len(y_test)):
    plt.plot([i, i], [y_test_sorted[i], y_hat_hat[i]], 'b-')

plt.show()

# Calculate the variance and the standard deviation of the error
variance = np.var(y_test - y_hat)
std_dev = np.sqrt(variance)

print(f'Variance: {variance} Standard Deviation: {std_dev}')

# make a ribbon plot of expected values
plt.plot(y_test_sorted, 'r')
plt.fill_between(np.arange(len(y_test)), y_test_sorted - std_dev, y_test_sorted + std_dev, color='red', alpha=0.3)
plt.fill_between(np.arange(len(y_test)), y_test_sorted + std_dev, y_hat_hat , where = y_hat_hat > (y_test_sorted + std_dev), color='blue', alpha=0.3)
plt.fill_between(np.arange(len(y_test)), y_test_sorted - std_dev, y_hat_hat , where = y_hat_hat < (y_test_sorted - std_dev), color='blue', alpha=0.3)
plt.plot(y_hat_hat, 'b', linewidth = '0.5')
plt.show()


feature_weights = np.abs(w[1:])
# Get feature names
feature_names = train.columns[:-1]  # Assuming the last column is the target variable


# Plotting
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_weights, y=feature_names)
plt.xlabel('Coefficient Magnitude')
plt.ylabel('Features')
plt.title('Magnitude of Coefficients')
plt.show()

# plot the error distribution
error = y_test_sorted - y_hat_hat
sns.histplot(error, kde=True)
plt.xlabel('Error')
plt.ylabel('Density')
plt.title('Error Distribution')
plt.show()
