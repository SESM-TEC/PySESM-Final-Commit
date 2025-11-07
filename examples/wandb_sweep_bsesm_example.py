'''
Copyright (C) 2023-2025 Tecnológico de Costa Rica

This script demonstrates how to use Weights & Biases (wandb) Sweeps to perform
Bayesian hyperparameter optimization for a pysesm model.

Authors: The SESM Team 
License: 
'''

import logging
import torch
import wandb
import numpy as np

from pysesm.models.BSESM import BSESM, BSESMConfig
from pysesm.sparse_coding import ISTAConfig, StepSizeMethod
from pysesm.dictionaries import GaussianDictConfig, GaussianDictLayer
from pysesm.blocks.UniformPartitionManager import UniformPartitionConfig
from pysesm.utils.loggers import setup_logger
from pysesm.utils_dataset.generate_dataset import generate_gaussian_dataset
from pysesm.utils.metric_loggers import log_to_WB
from pysesm.utils_dataset.gaussian_covariance_density import generate_nondiag_covariance_matrices

# --- 1. SETUP LOGGER & DEVICE ---
logger = setup_logger(level=logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# --- 2. DEFINE THE WANDB SWEEP CONFIGURATION ---
# Here we specify the search strategy (bayes), the metric to optimize (mse_value),
# and the hyperparameters to search over with their respective ranges.
sweep_config = {
    'method': 'bayes',  # Bayesian optimization
    'metric': {
      'name': 'mse_value',
      'goal': 'minimize'   
    },
    'parameters': {
        'n_functions': {
            'distribution': 'q_uniform',
            'q': 1,
            'min': 10,
            'max': 80
        },
        'dict_alpha': {
            'distribution': 'log_uniform_values',
            'min': 1e-4,
            'max': 1e-2
        },
        'sc_lambd': {
            'distribution': 'log_uniform_values',
            'min': 1e-4,
            'max': 1e-2
        },
        'reg_gamma': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 5e-2
        },
        'weight_decay': {
            'distribution': 'q_uniform',
            'q': 5e-6,
            'min': 0,
            'max': 1e-2
        },
        'sc_epochs': {
            'distribution': 'q_uniform',
            'q': 5,
            'min': 10,
            'max': 250
        },
        'dict_epochs': {
            'distribution': 'q_uniform',
            'q': 5,
            'min': 100,
            'max': 500
        }
    }
}

# --- 3. PRE-GENERATE DATASET (to avoid regeneration in each run) ---
logger.info("Generating a fixed dataset for the sweep...")
sigma1, sigma2, sigma3 = generate_nondiag_covariance_matrices()
non_diag_sigmas = [sigma1, sigma2, sigma3]

(trainDataset, X_train, y_train,
 testDataset, X_test, y_test,
 gt_mu, gt_sigma) = generate_gaussian_dataset(
    n_samples=250,
    variances=non_diag_sigmas  
)

# --- 4. DEFINE THE TRAINING FUNCTION FOR A SINGLE RUN ---
def train():
    """
    This function is called by the wandb agent for each run of the sweep.
    It builds the model configuration from wandb.config, trains the model,
    and logs the results.
    """
    # Initialize a new wandb run
    run = wandb.init()
    cfg = wandb.config

    # --- Construct a technical run name ---
    # We can create a more informative name using the unique run ID and key hyperparameters.
    run_number = run.name.split('-')[-1]
    technical_name = f"run_{int(run_number):03d}-funcs_{cfg.n_functions}-sc_e_{cfg.sc_epochs}-dict_e_{cfg.dict_epochs}"
    wandb.run.name = technical_name
    # Deprecated: Save the new name to the summary, which is useful for tables in W&B reports
    # wandb.run.save() # This is not necessary any more
    
    # Access the hyperparameters for this run from wandb.config
    cfg = wandb.config
    logger.info(f"Starting run with config: {dict(cfg)}")

    # --- Build PySESM configs dynamically from the sweep config ---
    sparse_coding_config = ISTAConfig(
        epochs=cfg.sc_epochs,
        lambd=cfg.sc_lambd,
        step_size_method=StepSizeMethod.FROBENIUS,
        n_functions=cfg.n_functions,
        criterion=torch.nn.MSELoss(),
        device=device
    )

    dict_config = GaussianDictConfig(
        epochs=cfg.dict_epochs,
        alpha=cfg.dict_alpha,
        criterion=torch.nn.MSELoss(),
        optimizer_factory=lambda params, lr: torch.optim.AdamW(
            params, lr=lr, weight_decay=cfg.weight_decay
        ),
        regularization_func=GaussianDictLayer.electrostatic_regularization,
        regularization_gamma=cfg.reg_gamma,
        device=device
    )

    partition_config = UniformPartitionConfig(
        T=3,
        initial_bounds=torch.tensor([[-2, -2], [2, 2]], dtype=torch.float32),
        device=device
    )

    bsesm_config = BSESMConfig(
        n_features=2,
        model_epochs=100, # Fixed number of epochs for all runs
        sparse_coding_config=sparse_coding_config,
        dict_config=dict_config,
        partition_config=partition_config,
        log_interval=50, # Log less frequently during sweeps
        device=device
    )

    experiment = {
        "config": bsesm_config,
        "seed": 42,
        # The hook is now a simple lambda, its frequency is controlled by log_interval above.
        "sesm_hook": lambda info: log_to_WB(
            layer_name="EpochLosses", info={
                "sc_loss": info['sparse_coding_losses'][-1],
                "dict_loss": info['dictionary_losses'][-1]
            }
        ),        
    }

    # --- Instantiate and Train the Model ---
    model = BSESM(**experiment, logger=logger)
    
    try:
        model.partial_fit(X_train, y_train)
        
        # --- Evaluate and Log the Metric ---
        _, _, mse_value = model.performance_stats(X_test, y_test)
        
        logger.info(f"Run finished. Final MSE: {mse_value:.6f}")
        wandb.log({"mse_value": mse_value})

    except Exception as e:
        logger.error(f"Run failed with exception: {e}")
        # Log a high MSE value to penalize this run in the sweep
        wandb.log({"mse_value": 1e6})

# --- 5. START THE SWEEP ---
if __name__ == "__main__":
    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project="pysesm-hyperparameter-optimization")

    # Start the sweep agent
    logger.info(f"Starting wandb sweep agent with ID: {sweep_id}")
    # The 'count' argument specifies how many runs to execute.
    wandb.agent(sweep_id, function=train, count=200)
    logger.info("Sweep finished.")
