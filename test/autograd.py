import torch

#autograd video
x = torch.randn(3, requires_grad=True)

y = x + 2    # node + ; input : x and 2, output : y
#print(x)
#print(y)

# there is a backward and forward pass
# forward pass : compute the output of the node
# backward pass : compute the gradient of the node dy/dx 

z = y*y*2 # create a z node with y^^2 * 2 then dz/dy = 4y and dz/dx = dz/dy * dy/dx thus being 4y * 1 = 4y
z = z.mean()
z.backward()
#print(x.grad)


## prevent pytorch from tracking the gradient
# either you specify it when you create the tensor torch.randn(3, requires_grad=False)

# x.requires_grad_(False) # change the requires_grad attribute of the tensor
# x.detach() # create a new tensor that does not require gradient
# with torch.no_grad(): # context manager that disable gradient

#print(x)
#x.requires_grad_(False)
#y = x.detach()
with torch.no_grad():
    y = x + 2

    #print(y)

weights = torch.ones(4, requires_grad=True)
#print(weights)

for epoch in range(3):
    model_output = (weights*3).sum() # mo = 3 * w so dmo/dw = 3

    model_output.backward() # compute the gradient of mo with respect of w
    #print(weights.grad) # print the gradient of the weights
    weights.grad.zero_() # empty the gradient of the weights so it's not accumulated

# optimizer

weights = torch.ones(4, requires_grad=True)
print(weights)
optimizer = torch.optim.SGD([weights], lr=0.01) # optimizer that will update the weights
# using SGD - taking a small batch of the data and update the weights accordingly
optimizer.step() # update the weights
print(weights)

optimizer.zero_grad() # empty the gradient

