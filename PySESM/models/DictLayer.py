import numpy as np
import torch
import torch.nn.init as init
from tqdm import tqdm
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
    def __init__(self, n_samples, n_features, n_functions, psi, initialization):
        super().__init__()

        self.theta_size = int(n_features*(n_features+3)/2)

        self.n_samples = n_samples

        self.n_features = n_features

        self.n_functions = n_functions

        self.psi = psi

        if initialization == "Lecun":

            self.Theta = torch.nn.Parameter(torch.normal(mean=0, std=np.sqrt(1/self.theta_size), size=(self.theta_size, n_functions), requires_grad=True))

        elif initialization == "Xavier":

            self.Theta = torch.nn.Parameter(torch.normal(mean=0, std=np.sqrt(2/(self.theta_size + n_functions)), size=(self.theta_size, n_functions), requires_grad=True))

        elif initialization == "Prieto":

            self.Theta = self.initialization(self.theta_size, self.n_functions, self.n_features, 1e2, 1e3);

        else:

            self.Theta = torch.nn.Parameter(torch.normal(mean=0, std=0, size=(self.theta_size, self.n_functions), requires_grad=True))

        if initialization == "Lecun":

            self.dictionary = torch.normal(mean=0, std=np.sqrt(1/self.n_samples), size=(self.n_samples, self.n_functions))

        else:

            self.dictionary = torch.normal(mean=0, std=np.sqrt(2/self.n_samples), size=(self.n_samples, self.n_functions))

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

    def initialization(self, theta_size, n_functions, n_features, min_val, max_val):

        Theta = torch.nn.Parameter(torch.normal(mean=0, std=np.sqrt(1/self.theta_size), size=(self.theta_size, n_functions), requires_grad=True))
        # Theta = torch.empty(n_functions - n_features, n_functions)
        print("Features: ", n_features)
        print("Theta: ", Theta.shape)
        scaled_tensor = torch.rand(n_features, n_functions)

        # lower_bound = -np.sqrt(min_val) * 2/(self.theta_size + n_functions)
        # upper_bound = np.sqrt(max_val) * 2/(self.theta_size + n_functions)

        # sub_tensor_detached = Theta[-n_features:, :].mT.unsqueeze(2).detach()
        # sub_tensor_detached.uniform_(lower_bound, upper_bound)

        with torch.no_grad():
            Theta[-n_features:, :] = scaled_tensor
            print("First myu: ", scaled_tensor.shape)

        Q = generate_random_vectors(theta_size + n_features, max_val, min_val)

        Q = gram_schmidt(Q)

        #Un rango gigantesco
        D = torch.diag(torch.rand(theta_size + n_features) * (max_val - min_val) + min_val)

        Sigma = Q @ D @ Q.mT
        eigenvalues, _ = torch.linalg.eig(Sigma)
        # print("Eigenvalues S: ", eigenvalues)
        # print("Sigma: ", Sigma)

        L = torch.linalg.cholesky(Sigma).mT
        eigenvalues, _ = torch.linalg.eig(L)
        # print("Eigenvalues L: ", eigenvalues)
        # print("Cholesky: ", L)
        Rho = reshape_upper_triangle(get_upper_triangle(L), n_functions)
        print("Rho: ", Rho.shape)

        with torch.no_grad():
            Theta[:-n_features, :] = Rho
            # print("Rho: ", Rho)

        return Theta
