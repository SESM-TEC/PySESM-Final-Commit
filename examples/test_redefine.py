import torch

from pysesm.utils.gaussian_covariance_density import *
from pysesm.utils.mesh_generation import *
from pysesm.utils.design_matrices import *
from pysesm.base_functions.sub_block_partition import *
from pysesm.functions.GaussianApproximateSurrogateFunction import *
from pysesm.utils.plot_and_save_stats import *
from pysesm.utils.loggers import setup_logger
from pysesm.models.SESMS.SSESM import SSESM
from pysesm.models.BSESM.BSESM import BSESM

# Crear los directorios de resultados
os.makedirs('results_1/plots', exist_ok=True)
os.makedirs('results_1/stats', exist_ok=True)

N_iter = 5  #

experiment_1 = {
    "hyp_set": 1,
    "n_samples": 500,
    "n_features": 2,
    "l_functions": 100,
    "eig_range": [1e0, 1e1],
    "mu_range": [-2, 2],
    "vector_range": [1e0, 1e1],
    "ista_alpha": 0.05502,
    "ista_lambd": 0.01007,
    "dictionary_alpha": 0.08928,
    "rho_epochs": 5,
    "mu_epochs": 5,
    "model_epochs": 5,
    "dict_epochs": 5,
    "ista_epochs": 5,
    "T": 4,
    "weight_decay": 0.004875,
    "permutation_times": 1,
    "mode": "secuencial",
    "Seed": 45
}

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')

# Create a console handler
console_handler = logging.StreamHandler()

# Set the custom formatter for the console handler
console_handler.setFormatter(formatter)

# Set up the basic configuration
logging.basicConfig(level=logging.INFO, handlers=[console_handler])

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

xx_r, yy_r, zz_r = generate_random_samples(500, -2, 2, sigma_list, mu_list, experiment_1["Seed"])

# Dataset
data = []
trainDataset = {"X": xx_r.ravel(), "Y": yy_r.ravel(), "Z": zz_r.ravel()}
testDataset = {"X": xx.ravel(), "Y": yy.ravel(), "Z": zz.ravel()}
# Crear la matriz de diseño
X_train, y_train = create_design_matrix_train(xx_r, yy_r, zz_r, experiment_1)
# Crear la matriz de diseño
X_test, y_test = create_design_matrix_test(xx, yy, zz)


# Ejecutar el experimento
# Cambiar los datos por una matriz de matriz de diseño X
# Un condicional para una grafica de > dimensiones
# run_experiment deberia de estar fuera de esta clase

def run_experiment(X_train, y_train, X_test, y_test, hyperparams, model):
    T = hyperparams["T"]
    l_functions = hyperparams["l_functions"]
    seed = hyperparams["Seed"]
    weight_decay = hyperparams["weight_decay"]
    alpha = hyperparams["ista_alpha"]
    lambd = hyperparams["ista_lambd"]
    time, mse_value = 0, 0

    t, x_n = data_mapping(X_train, T)

    sub_blocks = locate_samples_in_sub_blocks(X_train, y_train, t, T)

    list_sub_blocks = generate_list_of_subblock(sub_blocks, l_functions, seed, weight_decay, alpha, lambd)

    normalize_sub_blocks(list_sub_blocks, T)
    model.partial_fit(list_sub_blocks, T)
    Z_predict, time, mse_value = model.performance_stats(X_test, y_test, list_sub_blocks)

    plot_surface(testDataset, X_train, y_train, Z_predict, experiment_1["hyp_set"], model.dfngroup, model.iter,
                 model.losses_ISTA, model.losses_Dictionary)

    return time, mse_value


# Instanciar la función de aproximación (surrogate function)
surrogate_function = GaussianApproximateSurrogateFunction(
    n_features=experiment_1["n_features"],
    n_functions=experiment_1["l_functions"],
    eig_range=experiment_1["eig_range"],
    mu_range=experiment_1["mu_range"],
    vector_range=experiment_1["vector_range"],
    seed=experiment_1["Seed"],
    logger=logger
)


# Crea una instancia de la clase SESMS
ssesm_model = SSESM(
    n_samples=experiment_1["n_samples"],
    n_features=experiment_1["n_features"],
    l_functions=experiment_1["l_functions"],
    eig_range=tuple(experiment_1["eig_range"]),
    mu_range=tuple(experiment_1["mu_range"]),
    vector_range=tuple(experiment_1["vector_range"]),
    model_epochs=experiment_1["model_epochs"],
    ista_epochs=experiment_1["ista_epochs"],
    rho_epochs=experiment_1["rho_epochs"],
    mu_epochs=experiment_1["mu_epochs"],
    ista_alpha=experiment_1["ista_alpha"],
    ista_lambd=experiment_1["ista_lambd"],
    dictionary_alpha=experiment_1["dictionary_alpha"],
    weight_decay=experiment_1["weight_decay"],
    surrogate_function=surrogate_function,
    permutation_times=experiment_1["permutation_times"],
    dfngroup=1,
    iter=0,
    seed=experiment_1["Seed"],
    T=experiment_1["T"],
    logger=logger,
debug=True
)

bsesm = BSESM(
    n_samples=experiment_1["n_samples"],
    n_features=experiment_1["n_features"],
    l_functions=experiment_1["l_functions"],
    eig_range=tuple(experiment_1["eig_range"]),
    mu_range=tuple(experiment_1["mu_range"]),
    vector_range=tuple(experiment_1["vector_range"]),
    model_epochs=experiment_1["model_epochs"],
    ista_epochs=experiment_1["ista_epochs"],
    rho_epochs=experiment_1["rho_epochs"],
    mu_epochs=experiment_1["mu_epochs"],
    ista_alpha=experiment_1["ista_alpha"],
    ista_lambd=experiment_1["ista_lambd"],
    dictionary_alpha=experiment_1["dictionary_alpha"],
    weight_decay=experiment_1["weight_decay"],
    surrogate_function=surrogate_function,
    dfngroup=1,
    iter=0,
    seed=experiment_1["Seed"],
    logger=logger,
    T=[experiment_1["T"], experiment_1["T"]],
    debug=True
)

# for i in range(N_iter):
#     ssesm_model.iter = i
#     # Ejecuta el experimento
#
#     time, mse = run_experiment(X_train, y_train, X_test, y_test, experiment_1, sesms_model)
#
#     # Almacena los resultados
#     data.append((i, time, mse))

# save_results(data=data, fngroup=1)

for i in range(N_iter):
    bsesm.iter = i
    # Ejecuta el experimento

    print(X_train, y_train)
    bsesm.partial_fit(X_train, y_train)
    Z_predict, time, mse_value = bsesm.performance_stats(X_test, y_test)

    plot_surface(testDataset, X_train, y_train, Z_predict, experiment_1["hyp_set"], bsesm.dfngroup, bsesm.iter,
                 bsesm.losses_ISTA, bsesm.losses_Dictionary)

    # Almacena los resultados
    data.append((i, time, mse_value))