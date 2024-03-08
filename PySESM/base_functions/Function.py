import torch
from PySESM.utils.linalg import to_triu_matrix, generate_random_vectors, get_upper_triangle

class Function:
    def __init__(self, n_features, n_functions):
        self.n_features = n_features
        self.n_functions = n_functions

class GaussianFunctions(Function):
    def __init__(self, n_features, n_functions, eig_range, mu_range):
        super().__init__(n_features, n_functions)
        self.eig_range = eig_range
        self.mu_range = mu_range
        self.theta_size = int(n_features*(n_features+3)/2)

    def initialize(self):
        """
        Here comes the initialization
        In every interface there comes an initialization
        for i in n_functions:
            generate_random_vectors
            gram_schmidt
            Sigma_inv
            A
            For each gaussian, Rho: 3
            For each gaussian, Myu: 2
            Min will be a vector and max will be a vector
        """
        Theta = torch.nn.Parameter(torch.normal(mean=0, std=np.sqrt(1/self.theta_size), size=(self.theta_size, n_functions), requires_grad=True))

        mu = torch.rand(n_features, n_functions) * (self.mu_range[1] - self.mu_range[0]) + self.mu_range[0]

        Rho = torch.zeros(self.theta_size - n_features, n_functions)

        for i in self.n_functions:
            Q = generate_random_vectors(n_features, self.eig_range[1], self.eig_range[0])
            Q = gram_schmidt(Q)
            D = torch.diag(torch.rand(n_features) * (self.eig_range[1] - self.eig_range[0]) + self.eig_range[0])
            Sigma = Q @ D @ Q.mT
            L = torch.linalg.cholesky(Sigma).mT
            rho = get_upper_triangle(L)
            for j in (self.theta_size - n_features):
                Rho[j, i] = rho[j]

        with torch.no_grad():
            Theta[:-n_features, :] = Rho
            Theta[-n_features:, :] = mu

        return Theta

    def __call__(self, x, Theta):
        # Toma los Rho del Theta que recibe
        rho = Theta[:-self.n_features, :]
        print("Rho: ", rho.shape)
        # Toma los Myu del Theta que recibe
        mu = Theta[-self.n_features:, :].mT.unsqueeze(2)
        print("Myu: ", mu.shape)
        # Toma los Rho y los representa como una matriz diagonal superior
        A = torch.stack([to_triu_matrix(rho[:, i]) for i in range(self.n_functions)], dim = 0)
        Sigma_inv = torch.matmul(A, A.mT)
        x_mu = x - mu
        print("X: ", x.shape)
        print("Sigma inverso: ", Sigma_inv.shape)
        print("X normalizado: ", x_mu.shape)

        exponent = -0.5 * torch.einsum('bij,bik,bji->jb', x_mu, Sigma_inv, x_mu.mT)
        print("Gaussiana: ", exponent.shape)
        result = torch.exp(exponent)

        return result
