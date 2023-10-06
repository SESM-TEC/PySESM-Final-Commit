import numpy as np
import torch


class DictLayer(torch.nn.Module):
    """
    A custom PyTorch module for implementing a dictionary layer with learnable parameters.

    This layer is designed for use in surrogate modeling and function approximation tasks.

    Args:
        n_features (int): The number of input features or dimensions.
        n_functions (int): The number of functions or basis functions.
        psi (callable): The function used for generating the layer's output.

    Attributes:
        theta_size (int): The size of the parameter tensor Theta, computed based on n_features.
        Theta (torch.nn.Parameter): Learnable parameter tensor for the layer's functions.
        n_samples (int): The number of samples (not used in this class).
        n_features (int): The number of input features or dimensions.
        n_functions (int): The number of functions or basis functions.
        psi (callable): The function used for generating the layer's output.
        dictionary (Tensor): The computed output of the layer.

    Methods:
        forward(x):
            Forward pass through the layer.
            Args:
                x (Tensor): Input data of shape (n_samples, n_features).
            Returns:
                Tensor: The computed output of the layer.

    Example:
        # Create a DictLayer instance
        model = DictLayer(n_features=2, n_functions=5, psi=your_psi_function)
        # Perform a forward pass
        output = model(input_data)
    """
    def __init__(self, n_features, n_samples, n_functions, psi):
        super().__init__()

        self.theta_size = int(n_features*(n_features+3)/2)
        self.Theta = torch.nn.Parameter(
            torch.normal(mean=0, std=1, size=(self.theta_size, n_functions), requires_grad=True))

        self.n_samples = n_samples

        self.n_features = n_features

        self.n_functions = n_functions

        self.psi = psi

        self.dictionary = torch.zeros((self.n_samples, self.n_functions))

    def forward(self, x):
      result = self.psi(x.mT,self.Theta)
      self.dictionary = result
      return torch.sum(result)
