import torch
import torch.nn as nn
import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
#print('softmax numpy:', outputs)

x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim=0)
#print(outputs)

# cross entropy is often used with softmax - and it works on multiple classes too
# while on Linear Regression, we were using SSE (Sum of Squared Errors)
# cross entropy is used for classification problems
# the loss is higher when the predicted probability is far from the actual label
# D(Y^, Y) = -1/N * sum(Y_i * log(Y^_i)) -> we want to minimize the cross entropy loss
# Y^_i is the predicted probability of the i-th class
# Y_i is the actual probability of the i-th class (0 or 1) - one-hot encoded, so for a 3 class problem, 
# it would be something like [0, 1, 0] for the second class
# we apply softmax on the output layer (Y^)

def cross_entropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss # / float(predicted.shape[0])

# y must be one-hot encoded
# if class 0: [1, 0, 0]
# if class 1: [0, 1, 0]
# if class 2: [0, 0, 1]
Y = np.array([1, 0, 0])

# y_pred has probabilities
Y_pred_good = np.array([0.7, 0.2, 0.1])
Y_pred_bad = np.array([0.1, 0.3, 0.6])
l1 = cross_entropy(Y, Y_pred_good)
l2 = cross_entropy(Y, Y_pred_bad)
# print(f'Loss1 numpy: {l1:.4f}')
# print(f'Loss2 numpy: {l2:.4f}')

loss = nn.CrossEntropyLoss()
# carful: cross entropy from nn applies
# nn.LogSoftmax + nn.NLLLoss (Negative Log Likelihood Loss)
# so we don't need to apply softmax to the output layer
# Y has class labels, not one-hot encoded
# Y_pred has raw scores (logits), not Softmax

# 3 samples
Y = torch.tensor([2, 0, 1])

# nsamples x nclasses = 3 x 3
Y_pred_good = torch.tensor([[0.1, 1.0, 2.1], [2.0, 1.0, 0.1], [0.1, 3.0, 0.1]])
Y_pred_bad = torch.tensor([[2.5, 1.0, 0.3], [0.5, 2.0, 2.3], [0.5, 3.0, 0.3]])
l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)
print(l1.item())
print(l2.item())

_, predictions1 = torch.max(Y_pred_good, 1)
_, predictions2 = torch.max(Y_pred_bad, 1)
print(predictions1)
print(predictions2)