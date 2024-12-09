import torch
from torch.distributions.multivariate_normal import MultivariateNormal as MVN
import matplotlib.pyplot as plt


class MultivariateNormal:
    def __init__(self, n_features, mus, covariances, scale_factors):
        if n_features < 2:
            raise ValueError("The number of features must be greater than 1.")

        if len(mus) != len(covariances):
            raise ValueError("The number of means and covariances must be the same.")

        if len(mus) != len(scale_factors):
            raise ValueError("The number of means and scale_factors must be the same.")

        self.n_features = n_features
        self.mu = mus
        self.covariances = covariances
        self.scale_factors = scale_factors
        self.mvns = []

        for i in range(len(mus)):
            mu = mus[i]
            covariance = covariances[i]

            if mu.shape[0] != n_features:
                raise ValueError(
                    "The number of features of every mu must be the same as n_features."
                )

            self.mvns.append(MVN(mu, covariance))

    def sample_p(self, sample_points):
        pdf = torch.zeros(sample_points.shape[0], dtype=torch.float32)

        for i in range(len(self.mvns)):
            mvn = self.mvns[i]
            scale_factor = self.scale_factors[i].item()
            pdf += scale_factor * torch.exp(mvn.log_prob(sample_points))

        return pdf

    def sample_n(self, n_samples, search_space):
        # min_separation = 1

        X = torch.tensor([], dtype=torch.float32)

        for i in range(self.n_features):
            a = search_space[0]
            b = search_space[1]
            # x = torch.rand(n_samples) * (b - a) + a
            x = torch.linspace(a, b, n_samples)
            X = torch.cat([X, x.unsqueeze(1)], dim=1)

        # mesh_grids = torch.meshgrid([X[:, n] for n in range(self.n_features)])

        # X = torch.stack([mesh_grids[n].ravel() for n in range(self.n_features)], dim=1)
        y = self.sample_p(X)

        # total_points = X.shape[0]

        # selected_indexes = []

        # while len(selected_indexes) < n_samples:
        #     random_index = torch.randint(0, total_points, (1,)).item()

        #     if all(abs(random_index - existing_index) >= min_separation for existing_index in selected_indexes):
        #         selected_indexes.append(random_index)

        # sampled_indices = selected_indexes

        # X = torch.stack([X[sampled_indices, n] for n in range(self.n_features)], dim=1)
        # y = y[sampled_indices]

        return X, y

    def plot(self, n_samples, samples, savefig=False, filepath=None):
        n_plots = 4
        plot_elevs = [30, 60, 90, 30]
        plot_azims = [30, 60, 90, 120]

        # grids = torch.meshgrid(*samples)

        # xy_grid = torch.stack([grids[n].ravel() for n in range(self.n_features)], dim=1)

        # pdf_values = self.sample_p(xy_grid).reshape((n_samples,) * self.n_features)

        pdf_values = self.sample_p(samples)

        if self.n_features == 2:
            # X = grids[0]
            # Y = grids[1]

            X = samples[:, 0]
            Y = samples[:, 1]
        else:
            # reduced_xy_grid = self.pca(xy_grid, 2)

            reduced_xy_grid = self.pca(samples, 2)

            # X = reduced_xy_grid[:, 0].reshape((n_samples,) * self.n_features)
            # Y = reduced_xy_grid[:, 1].reshape((n_samples,) * self.n_features)
            X = reduced_xy_grid[:, 0]
            Y = reduced_xy_grid[:, 1]

        fig = plt.figure(figsize=(8, 8))

        for i in range(n_plots):
            ax = fig.add_subplot(2, 2, i + 1, projection="3d")
            ax.scatter(
                X.numpy(),
                Y.numpy(),
                pdf_values.numpy(),
                c=pdf_values.numpy(),
                cmap="plasma",
            )
            ax.view_init(elev=plot_elevs[i], azim=plot_azims[i])

        plt.tight_layout()

        if savefig:
            plt.savefig(filepath)

        plt.show()

    def pca(self, X, n_components=2):
        torch.manual_seed(1024)

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
