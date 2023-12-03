import torch
from torch.distributions.multivariate_normal import MultivariateNormal as MVN
import matplotlib.pyplot as plt

class MultivariateNormal:
    def __init__(self, n_features, mus, covariances, scale_factors):
        if n_features < 2:
            raise ValueError('The number of features must be greater than 1.')
        
        if len(mus) != len(covariances):
            raise ValueError('The number of means and covariances must be the same.')        
        
        if len(mus) != len(scale_factors):
            raise ValueError('The number of means and scale_factors must be the same.')
        
        self.n_features = n_features
        self.mu = mus
        self.covariances = covariances
        self.scale_factors = scale_factors
        self.mvns = []
        
        for i in range(len(mus)):
            mu = mus[i]
            covariance = covariances[i]
            
            if mu.shape[0] != n_features:
                raise ValueError('The number of features of every mu must be the same as n_features.')
            
            self.mvns.append(MVN(mu, covariance))

        
    def sample_p(self, sample_points):
        pdf = torch.zeros(sample_points.shape[0], dtype=torch.float32)
        
        for i in range(len(self.mvns)):
            mvn = self.mvns[i]
            scale_factor = self.scale_factors[i]
            pdf += scale_factor * torch.exp(mvn.log_prob(sample_points))
        
        return pdf
    
    
    def sample_n(self, n_samples):
        min_separation = 1
        
        X = torch.tensor([], dtype=torch.float32)

        for i in range(self.n_features):
            x = torch.linspace(-2, 2, 100)
            X = torch.cat([X, x.unsqueeze(1)], dim=1)
            
        mesh_grids = torch.meshgrid([X[:, n] for n in range(self.n_features)])

        X = torch.stack([mesh_grids[n].ravel() for n in range(self.n_features)], dim=1)
        y = self.sample_p(X)
        
        total_points = X.shape[0]

        selected_indexes = []

        while len(selected_indexes) < n_samples:
            random_index = torch.randint(0, total_points, (1,)).item()

            if all(abs(random_index - existing_index) >= min_separation for existing_index in selected_indexes):
                selected_indexes.append(random_index)

        sampled_indices = selected_indexes

        X = torch.stack([X[sampled_indices, n] for n in range(self.n_features)], dim=1)
        y = y[sampled_indices]
        
        return X, y
        
    
    def plot(self, n_samples):
        n_plots = 4
        plot_elevs = [30, 60, 90, 30]
        plot_azims = [30, 60, 90, 120]
        
        samples = torch.tensor([])
        
        for i in range(self.n_features):
            feature = torch.linspace(-2, 2, n_samples)
            samples = torch.cat([samples, feature.unsqueeze(1)], dim=1)
            
        fig = plt.figure(figsize=(8, 8))
        
        if(self.n_features == 2):
            X, Y = torch.meshgrid(samples[:, 0], samples[:, 1])
            
            xy_grid = torch.stack([X.ravel(), Y.ravel()], dim=1)
            
            pdf_values = self.sample_p(xy_grid).reshape(n_samples, n_samples)
                        
            for i in range(n_plots):
                ax = fig.add_subplot(2, 2, i+1, projection='3d')
                ax.plot_surface(X.numpy(), Y.numpy(), pdf_values.numpy(), cmap='plasma')
                ax.view_init(elev=plot_elevs[i], azim=plot_azims[i])
        else:
            pdf_values = self.sample_p(samples)
            
            samples = self.pca(samples)
            
            X = samples[:, 0]
            Y = samples[:, 1]
            
            for i in range(n_plots):
                ax = fig.add_subplot(2, 2, i+1, projection='3d')
                ax.scatter(X.numpy(), Y.numpy(), pdf_values.numpy(), c=pdf_values.numpy(), cmap='plasma')
                ax.view_init(elev=plot_elevs[i], azim=plot_azims[i])
            
        # Show the plot
        plt.tight_layout()
        plt.show()
                
        
    def pca(self, X, n_components=2):
        mean = torch.mean(X, dim=0)
        std = torch.std(X, dim=0)

        # Standardize the data
        X_std = (X - mean) / std

        # Step 2 & 3: Compute SVD
        U, S, V = torch.pca_lowrank(X_std)

        # Step 4: Select the number of principal components
        U = U[:, :n_components]

        # Step 5: Project data onto lower-dimensional space
        X_reduced = torch.mm(U, torch.diag(S[:n_components]))
        
        return X_reduced