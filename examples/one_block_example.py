import logging
import torch

from pysesm.enums import SurrogateFunctionEnum
from pysesm.models import BSESM, SSESM, SESM
from pysesm.utils.loggers import setup_logger
from pysesm.utils.generate_dataset import generate_gaussian_dataset, generate_one_gaussian_dataset
from pysesm.utils.plot_and_save_stats import plot_surface

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# SESM CONFIGURATION
experiment = {
    "hyp_set": 1,
    "n_samples": 500,
    "n_features": 2,
    "n_functions": 1,
    "eig_range": [1.0e0, 1.0e1],
    "mu_range": [-2.0, 2.0],
    "ista_alpha": 0.1,
    "ista_lambd": 0.0,
    "dictionary_alpha": 0.3,
    "rho_epochs": 10,
    "mu_epochs": 10,
    "model_epochs": 2500,
    "dict_epochs": 5,
    "ista_epochs": 10,
    "psi": SurrogateFunctionEnum.GAUSSIAN,
    "T": 1,
    "initial_bounds": torch.tensor([[-2, -2], [2, 2]], dtype=torch.float32),
    "weight_decay": 0.0,
    "permutation_times": 1,
    "seed": 45,
    "dfngroup": 1,
    "iter": 0,
    "debug": True,
}

def show_data(X,y,c,marker,label,ax=None):
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    

    # Plot training data
    ax.scatter(X[:, 0], X[:, 1], y, 
               c=c, marker=marker, label=label)
    
    ax.set_xlabel('x_1')
    ax.set_ylabel('x_2')
    ax.set_zlabel('y')
    ax.legend()

    plt.show(block=False)
    return ax

# DATA GENERATION
trainDataset, X_train, y_train, testDataset, X_test, y_test = generate_gaussian_dataset(experiment)

# ax = show_data(X_train,y_train,'r','x','Training')
# show_data(X_test,y_test,'0.4','.','Test',ax)

# RESULTS FOLDER NAME CREATION
folder_name = f"results_one_block_{experiment['hyp_set']}"

# LOGGER INSTANCE
logger = setup_logger()

# INSTANTIATE THE MODELS
ssesm_model = SSESM(**experiment,logger=logger)
#bsesm_model = BSESM(**experiment,logger=logger)

try:
    # TRAIN AND TEST THE ALL MODELS
    for model in [ssesm_model]: # bsesm_model
        logging.info("Training model {}".format(model.__class__.__name__))
        model_folder = f"{folder_name}_{model.__class__.__name__}"
        model.partial_fit(X_train, y_train)
        Z_predict, time, mse_value = model.performance_stats(X_test, y_test)

        logging.info("Model: {}, MSE Value = {:.6f}, time ={:.6f}".format(model.__class__.__name__, mse_value, time))

        plot_surface(testDataset, X_train, y_train, Z_predict, model, experiment["hyp_set"])

    plt.show(block=True)
except KeyboardInterrupt:
    print("\nShutting down...")
    plt.close('all')
    exit(0)
