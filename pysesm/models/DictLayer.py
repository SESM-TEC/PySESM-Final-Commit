import torch

from pysesm.functions.ApproximateSurrogateFunction import ApproximateSurrogateFunction


class DictLayer(torch.nn.Module):
    """
    A custom PyTorch module for implementing a dictionary layer with learnable parameters.

    This layer is designed for use in surrogate modeling and function approximation tasks.

    Attributes:
        psi (ApproximateSurrogateFunction): The function used for generating the layer's output.
        theta_parameter_vector (torch.nn.Parameter): Learnable parameter tensor for the layer's functions.
        dictionary (torch.Tensor): The computed output of the layer.
        n_samples (int): The number of samples (not used in this class).
        n_features (int): The number of input features or dimensions.
        n_functions (int): The number of functions or basis functions.
        losses (list): Losses history if log_losses parameter is set to ture
        criterion (torch.nn.modules.loss._Loss): Loss function used to compute the loss of the model.
        optimizer (torch.optim.Optimizer): Preferred optimizer used in the training of the model.
    """

    def __init__(self, n_samples: int, psi: ApproximateSurrogateFunction, alpha: float,
                 criterion: torch.nn.modules.loss._Loss = None, optimizer: torch.optim.Optimizer = None):
        """
        Method

        Args:
            n_samples:
            psi:
            alpha:
            criterion:
            optimizer:
        """
        super(DictLayer, self).__init__()

        self.n_samples = n_samples
        self.alpha = alpha
        
        self.psi = psi
        
        self.n_features = self.psi.n_features
        self.n_functions = self.psi.n_functions
        self.losses = []

       
        self.theta_parameter_vector = self.psi.initialize()

        self.dictionary = None

        if criterion is None:
            self.criterion = torch.nn.MSELoss()
        else:
            self.criterion = criterion

        if optimizer is None:
            self.optimizer = torch.optim.SGD(self.parameters(), lr=alpha)
        else:
            self.optimizer = optimizer

    # def fit(self, X, y, epochs, h, alpha, rho_flag, mu_flag, log_losses=True):
    #     """
    #     Let's train the model.
    #
    #     Args:
    #         X (Tensor): Input data of shape (n_samples, n_features).
    #         y (Tensor): Target data of shape (n_samples,).
    #         epochs (int): Number of training epochs.
    #         h (Tensor): Sparse vector for selecting basis functions.
    #         alpha (float): Learning rate for optimization.
    #         log_losses (bool): Whether to log losses during training (default True).
    #     """
    #     optimizer = torch.optim.SGD(self.parameters(), lr=alpha)
    #     criterion = torch.nn.MSELoss()
    #
    #     # for i in tqdm(range(epochs), desc='Training dictionary'):
    #     for _ in range(epochs):
    #         self.forward(X, rho_flag, mu_flag)
    #         y_pred = self.dictionary @ h
    #
    #         loss = criterion(y_pred, y)
    #         optimizer.zero_grad()
    #
    #         loss.backward(retain_graph=True)
    #         optimizer.step()
    #
    #         if log_losses:
    #             self.losses.append(loss.item())
    #
    # def forward_old(self, x, rho_flag: bool, mu_flag: bool):
    #     """
    #
    #     Args:
    #         x:
    #         rho_flag:
    #         mu_flag:
    #
    #     Returns:
    #
    #     """
    #     result = self.psi(x.mT, self.theta_parameter_vector, rho_flag, mu_flag)
    #     # Parameter readjust must happen outside this function as it should return the new weights
    #     self.dictionary = result
    #     # RETURN VALUE NEVER USED
    #     return torch.sum(result)

    def  initialize_layer(self, X: torch.Tensor) -> None:
        """
        Method that initializes the dictionary of the layer. It needs a input value to do so, so it can be done at __init__.

        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features).
        """
        self.dictionary = self.psi(X.mT, self.theta_parameter_vector)

    def partial_fit(self, X: torch.Tensor, y: torch.Tensor, h: torch.Tensor, epochs: int, rho_flag: bool = False,
                    mu_flag: bool = False, log_losses: bool = True) -> None:
        """
        Method that does a partial fit on the model without redefining the weights of its layers.

        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features).
            y (torch.Tensor): Target data of shape (n_samples,).
            h (torch.nn.Parameter): Sparse vector which holds the dictionary weights for the target function
            epochs (int): Number of training epochs for the layer.
            rho_flag (bool): Whether the rho values will be updated or not.
            mu_flag (book): Whether the mu values will be updated or not.
            log_losses (bool): Whether the losses of each epoch should be recorded.
        """

        for _ in range(epochs):
            self.dictionary = self.forward(X, rho_flag, mu_flag)
            y_pred = self.dictionary @ h
            loss = self.criterion(y_pred, y)
            self.optimizer.zero_grad()

            loss.backward(retain_graph=True)
            self.optimizer.step()

            if log_losses:
                self.losses.append(loss.item())

    def forward(self, X: torch.Tensor, rho_flag: bool = False, mu_flag: bool = False):
        """
        Method that computes the forward pass for the current layer.

        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features).
            rho_flag (bool): Whether the rho values will be updated or not.
            mu_flag (book): Whether the mu values will be updated or not.

        Returns:
            torch.Tensor: Input values evaluated using the psi function.
        """
        return self.psi(X.mT, self.theta_parameter_vector, rho_flag, mu_flag)
