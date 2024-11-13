import torch
import torch.nn as nn

# activation functions:
# apply a non linear transformation and decide whether a neuron should be activated or not.
# why do we need activation functions?
# for ex -  wT .x = y 
# if we don't have activation function, then the output will be a linear function of input - the model is a stacked linear regression model
# - not suited for complex problems
# after each layer, we apply an activation function to introduce non-linearity
# so -> f(wT .x) = y
# with non-linear transformations our network can learn better and perform more complex tasks
# after each layer we typically use an activation function
# some activation functions:
# 1 binary step function
# 2 sigmoid
# 3 tanh
# 4 ReLU
# 5 Leaky Re
# 6 Softmax

# binary step function
# f(x) = 1 if x >= threshold, 0 otherwise -> not used in practice

# sigmoid
# f(x) = 1 / (1 + e^-x) -> squashes numbers between 0 and 1 -> typically used in output layer of a binary classification

# tanh
# f(x) = 2 / (1 + e^-2x) -1 -> squashes numbers between -1 and 1 -> typically used in hidden layers

# ReLU
# f(x) = max(0, x) -> O for negatives values, and input as output for positive values -> most widely used activation function -> computationally efficient
# this is very popular -> if you don't know what to use, use ReLU for hiddel layers