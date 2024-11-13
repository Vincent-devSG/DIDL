import torch
import torch.nn as nn

# multiclass problem
class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # no softmax at the end
        return out
model = NeuralNet2(input_size=28*28, hidden_size=5, num_classes=3)
criterion = nn.CrossEntropyLoss() # applies softmax

# neural net with softmax
# which animal -> multiclass problem (dog or cat for example)
# input layer, then hidden layer with activation function like ReLU or tanh or sigmoid in the hidden layer
# then linear layer (output layer), then softmax (output layer) 
# with cross entropy loss() no need to apply softmax

# neural net with sigmoid
# binary classification problem (0 or 1) Is it a dog ? 
# input layer, then hidden layer with activation function like ReLU or tanh or sigmoid in the hidden layer
# then linear layer (output layer), then sigmoid (output layer)
# use nn.BCELoss() for binary classification - sigmoid at the end