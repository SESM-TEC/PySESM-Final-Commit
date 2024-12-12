import logging
import torch

from pysesm.enums import SurrogateFunctionEnum
from pysesm.models import BSESM, SSESM, SESM
from pysesm.utils.loggers import setup_logger
from pysesm.utils.generate_dataset import generate_gaussian_dataset
from pysesm.utils.plot_and_save_stats import plot_surface

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# SESM CONFIGURATION
experiment = {
    "hyp_set": 1,
    "n_samples": 500,
    "n_features": 2,
    "n_functions": 25,
    "eig_range": [1.0e0, 1.0e1],
    "mu_range": [0.0, 1.0],
    "vector_range": [1e0, 1e1],
    "ista_alpha": 0.07,
    "ista_lambd": 0.01007,
    "dictionary_alpha": 0.07007,
    "rho_epochs": 5,
    "mu_epochs": 5,
    "model_epochs": 5,
    "dict_epochs": 5,
    "ista_epochs": 5,
    "psi": SurrogateFunctionEnum.GAUSSIAN,
    "T": 1,
    "initial_bounds": torch.tensor([[-2, -2], [2, 2]], dtype=torch.float32),
    "weight_decay": 0.004875,
    "permutation_times": 1,
    "seed": 45,
    "dfngroup": 1,
    "iter": 0,
    "debug": True,
}

# DATA GENERATION
trainDataset, X_train, y_train, testDataset, X_test, y_test = generate_gaussian_dataset(experiment)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot training data
ax.scatter(X_train[:, 0], X_train[:, 1], y_train, 
          c='r', marker='x', label='Training')

# Optionally plot test data too
ax.scatter(X_test[:, 0], X_test[:, 1], y_test, 
          c='0.4', marker='.', label='Test')

ax.set_xlabel('x_1')
ax.set_ylabel('x_2')
ax.set_zlabel('y')
ax.legend()

# RESULTS FOLDER NAME CREATION
folder_name = f"results_one_block_{experiment['hyp_set']}"

# LOGGER INSTANCE
logger = setup_logger()

# INSTANTIATE THE MODELS
ssesm_model = SSESM(**experiment,logger=logger)

bsesm_model = BSESM(**experiment,logger=logger)


# TRAIN AND TEST THE ALL MODELS
for model in [bsesm_model, ssesm_model]:
    logging.info("Training model {}".format(model.__class__.__name__))
    model_folder = f"{folder_name}_{model.__class__.__name__}"
    model.partial_fit(X_train, y_train)
    Z_predict, time, mse_value = model.performance_stats(X_test, y_test)

    logging.info("Model: {}, MSE Value = {:.6f}, time ={:.6f}".format(model.__class__.__name__, mse_value, time))

    plot_surface(testDataset, X_train, y_train, Z_predict, model_folder, model, experiment["hyp_set"])

plt.show()