import numpy as np
import torch
import torch.nn.init as init
from tqdm import tqdm
from sklearn.decomposition import PCA

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
    def __init__(self, n_samples, n_features, n_functions, psi, initialization):
        super().__init__()

        self.theta_size = int(n_features*(n_features+3)/2)

        if initialization == "Lecun":

            self.Theta = torch.nn.Parameter(torch.normal(mean=0, std=np.sqrt(1/self.theta_size), size=(self.theta_size, n_functions), requires_grad=True))

        elif initialization == "Xavier":

            self.Theta = torch.nn.Parameter(torch.normal(mean=0, std=np.sqrt(2/self.theta_size), size=(self.theta_size, n_functions), requires_grad=True))

        elif initialization == "Prieto":

            self.Theta = self.initialize_Theta_uniform(self.theta_size, n_functions, 1, 1, 0, initialization="He");
            # self.Theta = self.initialize_Theta_normal(self.theta_size, n_functions);
            # Theta = torch.empty(self.theta_size, n_functions)
            # init.xavier_uniform_(Theta)
            # Theta = (Theta - Theta.min()) / (Theta.max() - Theta.min())
            # self.Theta = torch.nn.Parameter(Theta, requires_grad=True)

        else:

            self.Theta = torch.nn.Parameter(torch.normal(mean=0, std=0, size=(self.theta_size, n_functions), requires_grad=True))

        self.n_samples = n_samples

        self.n_features = n_features

        self.n_functions = n_functions

        self.psi = psi

        if initialization == "Prieto":

            self.dictionary = torch.normal(mean=0, std=np.sqrt(1/self.n_samples), size=(self.n_samples, self.n_functions))

        else:

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
      result = self.psi(x.mT,self.Theta)
      self.dictionary = result
      return torch.sum(result)

    def initialize_Theta_uniform(self, theta_size, n_functions, min_val, max_val, mean, initialization="Xavier"):
        if initialization == "Xavier":
            variance = 2.0 / (theta_size + n_functions)
        elif initialization == "He":
            variance = 2.0 / theta_size
        else:
            variance = 1.0 / theta_size

        std_dev = np.sqrt(variance)
        Theta = torch.nn.Parameter(torch.normal(mean=mean, std=std_dev, size=(self.theta_size, n_functions), requires_grad=True))

        lower_bound = -np.sqrt(min_val) * std_dev
        upper_bound = np.sqrt(max_val) * std_dev
        with torch.no_grad():
            Theta.uniform_(lower_bound, upper_bound)

        Theta = torch.nn.Parameter(Theta, requires_grad=True)

        return Theta


    def initialize_Theta_normal(theta_size, n_functions):
        std = np.sqrt(2 / theta_size)
        theta = torch.rand(theta_size, n_functions)
        theta = std * (2 * theta - 1)
        Theta = torch.nn.Parameter(theta, requires_grad=True)
        return Theta
