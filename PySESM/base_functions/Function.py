import torch
from torch.distributions import Uniform
from PySESM.utils.linalg import to_triu_matrix


class Function:
    def __init__(self, n_features, n_functions):
        self.n_features = n_features
        self.n_functions = n_functions

class GaussianFunctions(Function):
    def gaussian(self, x, Theta):

        rho = Theta[:-self.n_features, :]
        mu = Theta[-self.n_features:, :].mT.unsqueeze(2)
        low = 0
        high = 1
        dist = Uniform(low, high)
        uni_mu = dist.sample(mu.shape).to(mu.dtype)
        A = torch.stack([to_triu_matrix(rho[:, i]) for i in range(self.n_functions)], dim = 0)
        Sigma_inv = torch.matmul(A,A.mT)
        x_mu = x - uni_mu

        exponent = -0.5 * torch.einsum('bij,bik,bji->jb', x_mu, Sigma_inv, x_mu.mT)
        result = torch.exp(exponent)

        return result
