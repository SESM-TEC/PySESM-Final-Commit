import logging

from pysesm.functions import GaussianFunction
from pysesm.models import BSESM
from pysesm.models import SSESM
from pysesm.utils.generate_dataset import generate_gaussian_dataset
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
    "rho_epochs": 50,
    "mu_epochs": 50,
    "model_epochs": 50,
    "dict_epochs": 50,
    "ista_epochs": 50,
    "T": [1, 1],
    "initial_bounds": torch.tensor([[-2, -2], [2, 2]], dtype=torch.float32),
    "weight_decay": 0.004875,
    "permutation_times": 1,
    "mode": "batch",
    "Seed": 45,
}
# modes: sequential || batch

# DATA GENERATION
trainDataset, X_train, y_train, testDataset, X_test, y_test = generate_gaussian_dataset(
    experiment
)

# RESULTS FOLDER NAME CREATION
folder_name = f"results_{experiment["hyp_set"]}_{experiment["mode"]}"

# LOGGER INSTANCE
logger = setup_logger()

# INSTANTIATE THE SURROGATE FUNCTION TO BE USED BY THE MODEL
surrogate_function = GaussianFunction(
    n_features=experiment["n_features"],
    n_functions=experiment["n_functions"],
    logger=logger,
    eig_range=experiment["eig_range"],
    mu_range=experiment["mu_range"],
    vector_range=experiment["vector_range"],
    seed=experiment["Seed"],
)

# SELECT THE MODEL IMPLEMENTATION
model = None
if experiment["mode"] == "sequential":
    model = SSESM(
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
        initial_bounds=experiment["initial_bounds"],
    )
else:
    raise ValueError("Invalid model mode, must be 'sequential' or 'batch'")


# TRAIN AND TEST THE MODEL
for i in range(N_iter):
    model.iter = i

    model.partial_fit(X_train, y_train)
    Z_predict, time, mse_value = model.performance_stats(X_test, y_test)

    logging.info(
        "Iteration {}, MSE Value = {:.6f}, time ={:.6f}".format(i, mse_value, time)
    )

    plot_surface(
        testDataset,
        X_train,
        y_train,
        Z_predict,
        folder_name,
        model,
        experiment["hyp_set"],
    )
