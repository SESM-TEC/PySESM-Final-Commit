import torch
from PySESM.utils.linalg import to_triu_matrix


class Function:
    def __init__(self, n_features, n_functions):
        self.n_features = n_features
        self.n_functions = n_functions

class GaussianFunctions(Function):
    def gaussian(self, x, Theta):

        rho = Theta[:-self.n_features, :]
        mu = Theta[-self.n_features:, :].mT.unsqueeze(2)
        A = torch.stack([to_triu_matrix(rho[:, i]) for i in range(self.n_functions)], dim = 0)
        Sigma_inv = torch.matmul(A,A.mT)
        x_mu = x - mu

        exponent = -0.5 * torch.einsum('bij,bik,bji->jb', x_mu, Sigma_inv, x_mu.mT)
        result = torch.exp(exponent)

        return result
