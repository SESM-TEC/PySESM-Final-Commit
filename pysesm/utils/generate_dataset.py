from pysesm.utils.design_matrices import *
from pysesm.utils.gaussian_covariance_density import *
from pysesm.utils.mesh_generation import *


def generate_gaussian_dataset(experiemnt_data: dict):
    # Definicion de covarianzas no diagnonales
    sigma1, sigma2, sigma3 = generate_sigma_tensors()

    # Definicion de varianzas diagonales
    sigma1_d = 0.15 * torch.eye(2)
    sigma2_d = 0.2 * torch.eye(2)
    sigma3_d = 0.3 * torch.eye(2)

    mu1 = generate_mu(1, 1)
    mu2 = generate_mu(1, -1)
    mu3 = generate_mu(-1, -1)

    # sigmas Non-diagonal covariance
    sigma_list = [sigma1_d, sigma2_d, sigma3_d]

    mu_list = [mu1, mu2, mu3]

    xx, yy, zz = generate_mesh(50, -2, 2, sigma_list, mu_list)

    xx_r, yy_r, zz_r = generate_random_samples(
        500, -2, 2, sigma_list, mu_list, experiemnt_data["Seed"]
    )

    # Dataset
    data = []
    trainDataset = {"X": xx_r.ravel(), "Y": yy_r.ravel(), "Z": zz_r.ravel()}
    testDataset = {"X": xx.ravel(), "Y": yy.ravel(), "Z": zz.ravel()}
    # Crear la matriz de diseño
    X_train, y_train = create_design_matrix_train(xx_r, yy_r, zz_r, experiemnt_data)
    # Crear la matriz de diseño
    X_test, y_test = create_design_matrix_test(xx, yy, zz)

    return trainDataset, X_train, y_train, testDataset, X_test, y_test
