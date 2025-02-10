import torch
import matplotlib.pyplot as plt


def plot_sesm(sesm_model, samples, savefig=False, filepath=None):
    n_plots = 4
    plot_elevs = [30, 60, 90, 30]
    plot_azims = [30, 60, 90, 120]

    pdf_values = sesm_model.predict(samples).detach()

    if sesm_model.n_features == 2:

        X = samples[:, 0]
        Y = samples[:, 1]
    else:

        reduced_xy_grid = pca_sesm(samples, 2)

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


# TODO: Maybe plotting function must be abstracted outside of the class to preserve the Single Responsibility principle
def plot_sesm_loss(sesm_model, ylim=0.1, savefig=False, filepath=None):
    plt.plot(sesm_model.losses)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.ylim(0, ylim)

    if savefig:
        plt.savefig(filepath)

    plt.show()


# TODO: Cant figure out what does this function do
def pca_sesm(X: torch.Tensor, n_components: int = 2):
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
