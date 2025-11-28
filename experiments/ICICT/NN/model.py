"""ANN para regresion.

Simple feed-forward neural network used for regression experiments.
"""

import time

import torch
from torch import nn, optim


class NN(nn.Module):
    """Feed-forward neural network for regression tasks.

    Args:
        epochs (int): Number of training epochs.
        lr (float): Learning rate for the optimizer.
        hidden_dim (int): Hidden layer dimensionality.
        input_d (int): Input feature dimension.
    """

    def __init__(self, epochs, lr, hidden_dim, input_d):
        super().__init__()

        self.epochs = epochs
        self.lr = lr
        self.layers = nn.Sequential(
            nn.Linear(input_d, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return network predictions for input tensor `x`."""
        return self.layers(x)

    def train_nn(self, xtrain, ytrain, xtest, ytest):
        """Train the model using MSE loss and Adam optimizer.

        Returns:
            float: elapsed training time in seconds.
        """

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        print("\n Training NN...")

        start_time = time.time()
        for epoch in range(self.epochs):
            self.train()
            optimizer.zero_grad()

            # Forward pass on the training set
            predictions = self(xtrain)
            loss = criterion(predictions, ytrain.unsqueeze(1))

            # Backprop and optimizer step
            loss.backward()
            optimizer.step()

            # Evaluation on validation/test set (no grad)
            self.eval()
            with torch.no_grad():
                test_loss = criterion(self(xtest), ytest.unsqueeze(1))

            if (epoch + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{self.epochs}], "
                    f"mse_train: {loss.item():.6f}, "
                    f"mse_val: {test_loss.item():.6f}"
                )
        end_time = time.time()

        return end_time - start_time

    def test(self, xtest):
        """Run inference and return predictions as a detached CPU tensor."""
        print("\n Testing NN...")
        predictions = self.layers(xtest)
        predictions = predictions.detach().cpu()
        return predictions
