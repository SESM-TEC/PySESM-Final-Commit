from pysesm.utils.design_matrices import *
from pysesm.utils.gaussian_covariance_density import *
from pysesm.utils.mesh_generation import *


def generate_gaussian_dataset(experiment_config: dict):
    """
    Create a dataset from a weighted mixture of gaussians.
    It is fixed.  The means are fixed at (1,1), (1,-1) and (-1,-1),
    and the covariances are fixed to diagonal matrices with variances 0.15, 0.2 and 0.3.

    Args:
        experiment_data (dict): Configuration dictionary containing experiment parameters.
            Required keys:
                n_samples (int): Number of samples to generate
    
    Returns:
        tuple: A tuple containing:
            - trainDataset: Dictionary with keys X,Y,Z, with 500 random points
            - X_train (torch.Tensor): Training input features
            - y_train (torch.Tensor): Training target values
            - testDataset: The test dataset
            - X_test (torch.Tensor): Test input features
            - y_test (torch.Tensor): Test target values
    
    Raises:
        KeyError: If any required keys are missing from experiment_data
    """
    # Non-diagonal convariance matrices
    sigma1, sigma2, sigma3 = generate_nondiag_covariance_matrices()

    # Diagonal covariance matrices
    sigma1_d = 0.15 * torch.eye(2)
    sigma2_d = 0.2  * torch.eye(2)
    sigma3_d = 0.3  * torch.eye(2)

    # Define gaussian centers as tensors
    mu1 = generate_mu( 1,  1)
    mu2 = generate_mu( 1, -1)
    mu3 = generate_mu(-1, -1)

    # sigmas and mu finally used
    sigma_list = [sigma1_d, sigma2_d, sigma3_d]
    mu_list = [mu1, mu2, mu3]
    weights_list = [1.25,0.5,0.75]
    
    low_lim = -2
    high_lim = 2

    # Regular 2D cartesian grid with 50 division per dimension used for prediction
    xx, yy, zz = generate_mesh_samples(50, low_lim, high_lim, sigma_list, mu_list, weights_list)

    # Random samples used for training
    xx_r, yy_r, zz_r = generate_random_samples(
      experiment_config["n_samples"],
      low_lim, high_lim, 
      sigma_list, mu_list, weights_list, 
    )

    # Dataset
    data = []
    trainDataset = {"X": xx_r.ravel(), "Y": yy_r.ravel(), "Z": zz_r.ravel()}
    testDataset = {"X": xx.ravel(), "Y": yy.ravel(), "Z": zz.ravel()}
    # Crear la matriz de diseño
    X_train, y_train = create_design_matrix_train(xx_r, yy_r, zz_r, experiment_config)
    # Crear la matriz de diseño
    X_test, y_test = create_design_matrix_test(xx, yy, zz)

    return trainDataset, X_train, y_train, testDataset, X_test, y_test
