import torch
# TODO: UNUSED IMPORTS must be deleted
import numpy as np
from tqdm import tqdm

class ISTALayer(torch.nn.Module):
    """
    A custom PyTorch module for implementing a sparse vector layer with learnable parameters.

    This layer is designed for use in surrogate modeling and function approximation tasks.

    Attributes:
        n_functions (int): The number of functions or basis functions.
        h (torch.nn.Parameter): The sparse vector computed by the layer.
        losses (list): A list of the losses computed during training.

    Methods: TODO: Delete methods section as they should be documented at their declaration
        predict(x): TODO: No predict method, should it have it? And also the dictionary layer?
            Predicts the value of a function using a given dictionary.
            Args:
                dictionary (Tensor): A dictionary of shape (n_samples, n_functions).
            Returns:
                Tensor: The predicted values for each sample of the objective function.
    """
    def __init__(self, n_functions, seed, h=None):
        super().__init__()
        self.n_functions = n_functions
        torch.manual_seed(seed)

        self.h = torch.nn.Parameter(torch.rand(n_functions), requires_grad=True)
        self.h.data /= self.h.data.sum()

        self.losses = []


    def shrinkage(self, alpha, lambd) -> torch.Tensor:
        """
        Performs the shrinkage operation on the layer's parameters with the given hyperparameters.

        Args:
            alpha (float): Learning rate.
            lambd (float): Regularization parameter.
        Returns:
            torch.Tensor: The updated sparse vector.

        """
        return torch.sign(self.h) * torch.max(torch.abs(self.h) - alpha*lambd, torch.zeros_like(self.h))

    # TODO: Fit methods should redefine the weights, if continuous learning is wanted it should be called 'partial_fit'
    def fit(self, y, epochs, dictionary, alpha, lambd, weight_decay, log_losses=True) -> None:
        # TODO: Criterion should be passed as argument, recommendable to do it at the init method
        criterion = torch.nn.MSELoss()
        # TODO: Not sure if the optimizer should be passed as argument as well, because it receives arguments such as alpha and weight decay
        # TODO: Perhaps arguments such as alpha, weight_decay and others should be passed at init?
        optimizer = torch.optim.SGD(self.parameters(), lr=alpha, weight_decay=weight_decay)

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

    def forward(self):
        # TODO: forward method not implemented and it is defined in the torch.nn.Module Interface class. Should encapsulate
        # TODO: each epoch code in here.
        pass
