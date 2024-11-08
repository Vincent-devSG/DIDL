'''
import numpy as np

data = np.loadtxt('wine.csv')

# training loop

for epoch in range(1000):
    x, y = data
    # forward pass + backward + weight update

# this is BAD - time consuming and error prone

# Better way, is to give smaller sample of data, called batches of data

total_batches = 10
for epoch in range(1000):
    # loop over all batches
    for i in range(total_batches):
        x_batches, y_batches = ...
        # forward pass + backward + weight update

# this is better, but still not optimal

# Use DataSet and DataLoader to load the data - wine.csv in this case

'''

'''
    QUICK NOTES ABOUT BATCHES, DATASET AND DATALOADER:
    epoch = 1 forward, backward pass of ALL training samples
    batch_size = number of training samples in one forward, backward pass
    number of iterations = number of passes, each pass using [batch_size] number of samples

    e.g. 100 samples, batches_size = 20 --> 100/20 = 5 iterations for 1 epoch
    that means 1 epoch = 5 iterations to complete

'''

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class WineDataset(Dataset): # Inherit from Dataset class from PyTorch
    
    def __init__(self): # the constructor - loading the data
        # data loading
        xy = np.loadtxt('wine.csv', delimiter=',', dtype=np.float32, skiprows=1) # load the data - skip the first row
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]]) # n_samples, 1 - 2D array
        self.n_samples = xy.shape[0]

    def __getitem__(self, index): # call a precise item of the dataset with its index
        # dataset[0]
        return self.x[index], self.y[index]
    
    def __len__(self): # return the length of the dataset
        # len(dataset)
        return self.n_samples
    

def main():

    dataset = WineDataset()
    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=2) # num_workers = number of subprocesses to use for data loading

    # dataiter = iter(dataloader)
    # data = next(dataiter)

    # features, labels = data
    # print(features, labels)
    
    # training loop - dummy
    num_epochs = 2
    total_samples = len(dataset)
    n_iterations = math.ceil(total_samples / 4) # 4 is the batch size
    print(total_samples, n_iterations)

    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(dataloader):
            # forward, backward pass, update w
            if (i+1) % 5 == 0:
                print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}')
    
    torchvision.datasets.MNIST()
    # fashion-mnist, cifar, coco data set ....
    # --> next video is about dataset transformations - which is cool
    # Now we know how to make a dataset properly, by making a custom class inherited from Dataset class from PyTorch
    # We must implement the __getitem__ and __len__ methods
    # We can use DataLoader to load the data in batches - it is very useful for large datasets
    # Why Use Batches?
    # 1. Efficiency:
    #     - training on a single sample at a time is slow, especially on large datasets.
    #     - processing only small batches leverages
    #     - with batches, -> parallelization, GPU memory optimization, and vectorization
    # 2. Memory constraints:
    #     - loading the entire dataset into memory is not always possible - memory wise (would require too much)
    #     - batches allow us to load only the data we need for the current iteration, without using all the ressources
    # 3. Improved generalization:
    #    - smaller batches introduce a bit of "noise" in each update (small batches -> change in statisticall composition of the data)
    #    - this noise can help the model to generalize better

    # How the size of the batches affects the model?
    # - Smaller Batch Sizes (e.g., 16 or 32): Can lead to more "noisy" updates, which might improve generalization but slow down convergence.
    # - Larger Batch Sizes (e.g., 128, 256): Generally leads to smoother and faster convergence, but sometimes the model might overfit because each update is based on more uniform information.
    


if __name__ == "__main__":
    main()