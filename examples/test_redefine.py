import logging

from pysesm.functions import GaussianFunction
from pysesm.models import BSESM
from pysesm.models import SSESM
from pysesm.utils.design_matrices import *
from pysesm.utils.gaussian_covariance_density import *
from pysesm.utils.loggers import setup_logger
from pysesm.utils.mesh_generation import *
from pysesm.utils.plot_and_save_stats import *

N_iter = 1  #

experiment = {
    "hyp_set": 1,
    "n_samples": 500,
    "n_features": 2,
    "n_functions": 25,
    "eig_range": [1e0, 1e1],
    "mu_range": [0, 1],
    "vector_range": [1e0, 1e1],
    "ista_alpha": 0.07,
    "ista_lambd": 0.01007,
    "dictionary_alpha": 0.07007,
    "rho_epochs": 150,
    "mu_epochs": 150,
    "model_epochs": 150,
    "dict_epochs": 150,
    "ista_epochs": 150,
    "T": [2, 2],
    "initial_bounds": torch.tensor([[-2, -2], [2, 2]], dtype=torch.float32),
    "weight_decay": 0.004875,
    "permutation_times": 1,
    "mode": "sequential",
    "Seed": 45
}
# modes: sequential || batch

folder_name = f"results_{experiment["hyp_set"]}_{experiment["mode"]}"

logger = setup_logger()

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

xx_r, yy_r, zz_r = generate_random_samples(500, -2, 2, sigma_list, mu_list, experiment["Seed"])

# Dataset
data = []
trainDataset = {"X": xx_r.ravel(), "Y": yy_r.ravel(), "Z": zz_r.ravel()}
testDataset = {"X": xx.ravel(), "Y": yy.ravel(), "Z": zz.ravel()}
# Crear la matriz de diseño
X_train, y_train = create_design_matrix_train(xx_r, yy_r, zz_r, experiment)
# Crear la matriz de diseño
X_test, y_test = create_design_matrix_test(xx, yy, zz)

# Instanciar la función de aproximación (surrogate function)
surrogate_function = GaussianFunction(
    n_features=experiment["n_features"],
    n_functions=experiment["n_functions"],
    logger=logger,
    eig_range=experiment["eig_range"],
    mu_range=experiment["mu_range"],
    vector_range=experiment["vector_range"],
    seed=experiment["Seed"],
)

model = None
if experiment["mode"] == "sequential":
    model = SSESM(
        n_samples=experiment["n_samples"],
        n_features=experiment["n_features"],
        n_functions=experiment["n_functions"],
        model_epochs=experiment["model_epochs"],
        ista_epochs=experiment["ista_epochs"],
        rho_epochs=experiment["rho_epochs"],
        mu_epochs=experiment["mu_epochs"],
        ista_alpha=experiment["ista_alpha"],
        ista_lambd=experiment["ista_lambd"],
        dictionary_alpha=experiment["dictionary_alpha"],
        weight_decay=experiment["weight_decay"],
        surrogate_function=surrogate_function,
        permutation_times=experiment["permutation_times"],
        dfngroup=1,
        iter=0,
        seed=experiment["Seed"],
        T=experiment["T"],
        logger=logger,
        debug=True,
        initial_bounds=experiment["initial_bounds"],
        eig_range=tuple(experiment["eig_range"]),
        mu_range=tuple(experiment["mu_range"]),
        vector_range=tuple(experiment["vector_range"]),
    )
elif experiment["mode"] == "batch":
    model = BSESM(
        n_samples=experiment["n_samples"],
        n_features=experiment["n_features"],
        n_functions=experiment["n_functions"],
        eig_range=tuple(experiment["eig_range"]),
        mu_range=tuple(experiment["mu_range"]),
        vector_range=tuple(experiment["vector_range"]),
        model_epochs=experiment["model_epochs"],
        ista_epochs=experiment["ista_epochs"],
        rho_epochs=experiment["rho_epochs"],
        mu_epochs=experiment["mu_epochs"],
        ista_alpha=experiment["ista_alpha"],
        ista_lambd=experiment["ista_lambd"],
        dictionary_alpha=experiment["dictionary_alpha"],
        weight_decay=experiment["weight_decay"],
        surrogate_function=surrogate_function,
        dfngroup=1,
        iter=0,
        seed=experiment["Seed"],
        logger=logger,
        T=experiment["T"],
        debug=True,
        initial_bounds=experiment["initial_bounds"]
    )
else:
    raise ValueError("Invalid model mode, must be 'sequential' or 'batch'")

for i in range(N_iter):
    model.iter = i

    model.partial_fit(X_train, y_train)
    Z_predict, time, mse_value = model.performance_stats(X_test, y_test)

    logging.info("Iteration {}, MSE Value = {:.6f}, time ={:.6f}".format(i, mse_value, time))

    plot_surface(testDataset, X_train, y_train, Z_predict, folder_name, model, experiment["hyp_set"])

    # Almacena los resultados
    data.append((i, time, mse_value))
