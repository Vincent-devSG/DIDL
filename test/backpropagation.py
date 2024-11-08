import torch

# backpropagation video
# chain rule : dz/dx = dz/dy * dy/dx if y = f(x) and z = g(y) then dz/dx = g'(f(x)) * f'(x)

# computational graph
# z = x * y 
# multiplication operation is the node. f = x*y
# local gradients: dz/dx = y and dz/dy = x
# so dz/dx = dz/df * df/dx = 1 * y = y

# forward pass: compute the output of the node
# compute local gradients
# backward pass: compute the dLoss/dWeights using the chain rule

x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad=True)

# forward pass and compute the loss
y_hat = w * x 
loss = (y_hat - y)**2

print(loss)

# backward pass
loss.backward()
print(w.grad)

# update weights
# next forward and backward pass ...