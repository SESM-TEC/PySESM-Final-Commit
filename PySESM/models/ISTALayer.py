import torch
import numpy as np
from tqdm import tqdm

class ISTALayer(torch.nn.Module):
    """
    A custom PyTorch module for implementing a sparse vector layer with learnable parameters.

    This layer is designed for use in surrogate modeling and function approximation tasks.

    Args:
        n_functions (int): The number of functions or basis functions.

    Attributes:
        n_functions (int): The number of functions or basis functions.
        h (Tensor): The sparse vector computed by the layer.
        losses (list): A list of the losses computed during training.

    Methods:
        shrinkage(alpha, lambd):
            Performs the shrinkage operation on the layer's parameters.
            Args:
                alpha (float): The learning rate.
                lambd (float): The regularization parameter.
            Returns:
                Tensor: The updated sparse vector.
                
        predict(x):
            Predicts the value of a function using a given dictionary.
            Args:
                dictionary (Tensor): A dictionary of shape (n_samples, n_functions).
            Returns:
                Tensor: The predicted values for each sample of the objective function.
    """
    def __init__(self, n_functions):
        super().__init__()

        self.n_functions = n_functions
        self.h = torch.nn.Parameter(torch.ones((n_functions), requires_grad=True))
        
        self.losses = []
        
        
    def shrinkage(self, alpha, lambd):
        return torch.sign(self.h) * torch.max(torch.abs(self.h) - alpha*lambd, torch.zeros_like(self.h))


    def fit(self, y, epochs, dictionary, alpha, lambd, log_losses=True):
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=alpha, weight_decay=lambd)

        #for i in tqdm(range(epochs), desc='Training sparse vector'):
        for _ in range(epochs):
            y_pred = dictionary @ self.h
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        
            with torch.no_grad():
                self.h.data = self.shrinkage(alpha, lambd)
            
            if log_losses:
                self.losses.append(loss.item())



