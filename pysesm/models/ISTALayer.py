import torch


class ISTALayer(torch.nn.Module):
    """
    A custom PyTorch module for implementing a sparse vector layer with learnable parameters.

    This layer is designed for use in surrogate modeling and function approximation tasks.

    Attributes:
        n_functions (int): The number of functions or basis functions.
        h (torch.nn.Parameter): The sparse vector computed by the layer.
        losses (list): A list of the losses computed during training.

    """

    def __init__(self, n_functions: int, random_seed: int, weight_decay: float, alpha: float, lambd: float,
                 criterion=None, optimizer=None, h: torch.nn.Parameter = None):

        super().__init__()
        self.n_functions = n_functions
        self.random_seed = random_seed
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.lambd = lambd
        self.losses = []
        torch.manual_seed(random_seed)

        if h is None:
            self.initialize_h_vector()
        else:
            self.h = h

        if criterion is None:
            self.criterion = torch.nn.MSELoss()
        else:
            self.criterion = criterion

        if optimizer is None:
            self.optimizer = torch.optim.SGD(self.parameters(), lr=alpha, weight_decay=weight_decay)
        else:
            self.optimizer = optimizer(parameters=self.parameters(), lr=alpha, weight_decay=weight_decay)

    def initialize_h_vector(self) -> None:
        self.h = torch.nn.Parameter(torch.rand(self.n_functions), requires_grad=True)
        self.h.data /= self.h.data.sum()

    def shrinkage(self) -> torch.Tensor:
        """
        Performs the shrinkage operation on the layer's parameters with the given hyperparameters.

        Args:
            alpha (float): Learning rate.
            lambd (float): Regularization parameter.
        Returns:
            torch.Tensor: The updated sparse vector.

        """
        return torch.sign(self.h) * torch.max(torch.abs(self.h) - self.alpha * self.lambd, torch.zeros_like(self.h))

    def partial_fit(self, y, epochs, dictionary, log_losses=True) -> None:
        for _ in range(epochs):
            new_h = self.forward(y, dictionary, log_losses)
            if new_h is not None: self.h.data = new_h

    def forward(self, y, dictionary, log_losses=True):
        y_pred = dictionary @ self.h
        loss = self.criterion(y_pred, y)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

        if log_losses:
            self.losses.append(loss.item())

        with torch.no_grad():
            return self.shrinkage()
