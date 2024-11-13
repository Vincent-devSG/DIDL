# dataset = torchvision.datasets.MNIST(
#     root='data/', transform=torchvision.transforms.ToTensor())

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class WineDataset(Dataset): # Inherit from Dataset class from PyTorch
    
    def __init__(self, transform): # the constructor - loading the data
        xy = np.loadtxt('wine.csv', delimiter=',', dtype=np.float32, skiprows=1) # load the data - skip the first row
        self.n_samples = xy.shape[0]
        
        # we don't transform the data here to tensors, but allow the user to pass a transform ToTensor() that we will implement
        self.x = xy[:, 1:]
        self.y = xy[:, [0]]

        self.transform = transform

    def __getitem__(self, index): # call a precise item of the dataset with its index 
        sample =  self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def __len__(self): # return the length of the dataset
        # len(dataset)
        return self.n_samples

class ToTensor():
    def __call__(self, sample): # callable function of the class
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

class MulTransform():
    def __init__(self, factor):
        self.factor = factor
    
    def __call__(self, sample):
        input, target = sample
        input *= self.factor
        return input, target
        

dataset = WineDataset(transform=None)
first_data  = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features)


composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])
dataset = WineDataset(transform=composed)
first_data  = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features)


# --> understanding of the Dataset and DataLoader classes in PyTorch. Going back to DiveIntoDeepLearning. Closing the first chapter, Linear Regression.