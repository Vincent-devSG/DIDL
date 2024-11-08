import torch
import torch.nn as nn

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # defile the layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)
    
def train_model(model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    epochs: int = 1000,
    lr: float = 0.01,
    weight_decay: float = 0.0
    ) -> None:
    """
    Train the model.

    Args:
        model: the model
        x_train: the train set
        y_train: the true values
        x_test: the test set
        y_test: the true values
        epochs: the number of epochs
        lr: the learning rate

    Returns:
        None
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)   

    # Training loop
    for epoch in range(epochs):

        # Forward pass
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
    
        # Backward pass and optimization
        loss.backward()

        # Update the weights
        optimizer.step()

        # empty gradients - no accumulation
        optimizer.zero_grad()

        # Print progress every 100 epochs
        if (epoch + 1) % 100_000 == 0:
            print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')


    
