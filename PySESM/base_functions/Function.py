import torch
from PySESM.utils.linalg import to_triu_matrix

class Function:
    def __init__(self, n_features, n_functions):
        self.n_features = n_features
        self.n_functions = n_functions

class GaussianFunctions(Function):
    def gaussian(self, x, Theta):

        rho = Theta[:-self.n_features, :]
        print("Rho: ", rho.shape)
        mu = Theta[-self.n_features:, :].mT.unsqueeze(2)
        print("Myu: ", mu.shape)
        A = torch.stack([to_triu_matrix(rho[:, i]) for i in range(self.n_functions)], dim = 0)
        print("A: ", A.shape)
        eigenvalues, _ = torch.linalg.eig(A)
        # print("Eigenvalues A: ", eigenvalues)
        Sigma_inv = torch.matmul(A, A.mT)
        x_mu = x - mu
        print("X: ", x.shape)
        print("Sigma inverso: ", Sigma_inv.shape)
        eigenvalues, _ = torch.linalg.eig(Sigma_inv)
        # print("Eigenvalues S: ", eigenvalues)
        print("X normalizado: ", x_mu.shape)

        exponent = -0.5 * torch.einsum('bij,bik,bji->jb', x_mu, Sigma_inv, x_mu.mT)
        print("Gaussiana: ", exponent.shape)
        result = torch.exp(exponent)
        # print("Gaussian: ", result)

        return result
