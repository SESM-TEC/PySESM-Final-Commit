from tqdm import tqdm
import torch
import numpy as np
from sklearn.decomposition import PCA
from PySESM.utils.linalg import generate_random_vectors, gram_schmidt, get_upper_triangle, reshape_upper_triangle

class DictLayer(torch.nn.Module):
    """
    A custom PyTorch module for implementing a dictionary layer with learnable parameters.

    This layer is designed for use in surrogate modeling and function approximation tasks.

    Args:
        n_samples (int): The number of samples taken from the original function.
        n_features (int): The number of input features or dimensions.
        n_functions (int): The number of functions or basis functions.
        psi (callable): The function used for generating the layer's output.
        initialization (String): The initialization method to generate the paremeter tensor.

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
    def __init__(self, n_samples, psi):
        super().__init__()

        # No deberíamos pasar el funcional sino pasar el objeto entero

        self.n_samples = n_samples

        self.psi = psi

        #Matriz de n_features x n_features
        self.n_features = self.psi.n_features

        #N Gaussianas con una matriz de n_features x n_features
        self.n_functions = self.psi.n_functions

        self.Theta = self.psi.initialize()

        self.dictionary = torch.normal(mean=0, std=np.sqrt(1/self.n_samples), size=(self.n_samples, self.n_functions))

        self.losses = []


    def fit(self, X, y, epochs, h, alpha, log_losses=True):
        """
        Let's train the model.

        Args:
            X (Tensor): Input data of shape (n_samples, n_features).
            y (Tensor): Target data of shape (n_samples,).
            epochs (int): Number of training epochs.
            h (Tensor): Sparse vector for selecting basis functions.
            alpha (float): Learning rate for optimization.
            log_losses (bool): Whether to log losses during training (default True).
        """
        optimizer = torch.optim.SGD(self.parameters(), lr=alpha)
        criterion = torch.nn.MSELoss()

        #for i in tqdm(range(epochs), desc='Training dictionary'):
        for _ in range(epochs):
            self.forward(X)
            y_pred = self.dictionary @ h
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            if log_losses:
                self.losses.append(loss.item())


    def forward(self, x):
      result = self.psi(x.mT, self.Theta)
      self.dictionary = result
      return torch.sum(result)
