# 1 desing model (input, output size, forward pass)
# 2 Construct loss and optimizer
# 3 Training loop
#   - forward pass: compute prediction
#   - backward pass: gradients
#   - update weights
# Iterate over training loop

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 0 prepare data
# 1 model
# 2 loss and optim
# 3 training loop

# 0 - Data
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)
X = torch.from_numpy(X_numpy.astype(np.float32)) # convert to tensor but before to float32 cuz was double
y = torch.from_numpy(y_numpy.astype(np.float32))
# we want to reshape to have (100, 1) instad of (1, 100)
y = y.view(y.shape[0], 1) # view is a pytorch method to reshape like numpy reshape function

n_samples, n_features = X.shape

# 1 - Model
input_size = n_features
output_size = 1

model = nn.Linear(input_size, output_size) # setup the model Linear

# 2 - Loss and optim
lr = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# 3 - Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # forward pass
    y_predicted = model(X)
    loss = criterion(y_predicted, y)

    # backward pass
    loss.backward() # dLoss/dw

    # update (w)
    optimizer.step()

    # empty gradients - no accumulation
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

# plot the data and the model
predicted = model(X).detach().numpy() # detach to avoid tracking the gradients, so model(X).detach() is a new tensor, but grad_fn is None
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()