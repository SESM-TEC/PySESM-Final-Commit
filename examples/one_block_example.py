import logging
import torch

from pysesm.enums import SurrogateFunctionEnum
from pysesm.models import BSESM, SSESM, SESM
from pysesm.utils.loggers import setup_logger
from pysesm.utils.generate_dataset import generate_gaussian_dataset
from pysesm.utils.plot_and_save_stats import plot_surface

# SESM CONFIGURATION
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
    "rho_epochs": 5,
    "mu_epochs": 5,
    "model_epochs": 5,
    "dict_epochs": 5,
    "ista_epochs": 5,
    "surrogate_function": SurrogateFunctionEnum.GAUSSIAN,
    "T": 1,
    "initial_bounds": torch.tensor([[-2, -2], [2, 2]], dtype=torch.float32),
    "weight_decay": 0.004875,
    "permutation_times": 1,
    "Seed": 45
}

# DATA GENERATION
trainDataset, X_train, y_train, testDataset, X_test, y_test = generate_gaussian_dataset(experiment)

# RESULTS FOLDER NAME CREATION
folder_name = f"results_one_block_{experiment["hyp_set"]}"

# LOGGER INSTANCE
logger = setup_logger()

# INSTANTIATE THE MODELS
ssesm_model = SSESM(
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
    psi=experiment["surrogate_function"],
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

bsesm_model = BSESM(
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
    psi=experiment["surrogate_function"],
    dfngroup=1,
    iter=0,
    seed=experiment["Seed"],
    logger=logger,
    T=experiment["T"],
    debug=True,
    initial_bounds=experiment["initial_bounds"]
)


# TRAIN AND TEST THE ALL MODELS
for model in [bsesm_model, ssesm_model]:
    logging.info("Training model {}".format(model.__class__.__name__))
    model_folder = f"{folder_name}_{model.__class__.__name__}"
    model.partial_fit(X_train, y_train)
    Z_predict, time, mse_value = model.performance_stats(X_test, y_test)

    logging.info("Model: {}, MSE Value = {:.6f}, time ={:.6f}".format(model.__class__.__name__, mse_value, time))

    plot_surface(testDataset, X_train, y_train, Z_predict, model_folder, model, experiment["hyp_set"])
