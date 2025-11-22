"""
N-Dimensions Example.

Demonstrates the capability of the SESM framework to approximate functions
in higher-dimensional input spaces (N > 2).

Copyright (c) 2023-2025, Tecnológico de Costa Rica
All rights reserved.

This source code is licensed under the BSD 3-Clause License found in the
LICENSE file in the root directory of this source tree.

SPDX-License-Identifier: BSD-3-Clause
"""

import logging
import torch

from pysesm.models.SSESM import SSESM, SSESMConfig
from pysesm.sparse_coding import ISTAConfig, StepSizeMethod
from pysesm.dictionaries import GaussianDictConfig, GaussianDictLayer
from pysesm.blocks.UniformPartitionManager import UniformPartitionConfig
from pysesm.utils.loggers import setup_logger
from pysesm.utils_dataset.generate_dataset import generate_custom_nd_function_dataset
from pysesm.utils_dataset.distribution_functions import nd_paraboloid

# --- Main Script ---

# 1. SETUP LOGGER
logger = setup_logger(level=logging.INFO)

# 2. DEFINE EXPERIMENT PARAMETERS
n_features = 4  # <-- Key parameter: Number of input dimensions
n_functions = 50
n_samples = 2000

# 3. DEFINE MODEL CONFIGURATIONS

sparse_coding_config = ISTAConfig(
    epochs=100,
    alpha=0.1,
    lambd=0.001,
    step_size_method=StepSizeMethod.FROBENIUS,
    n_functions=n_functions,
    criterion=torch.nn.MSELoss(),
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

dict_config = GaussianDictConfig(
    epochs=50,
    alpha=0.01,
    criterion=torch.nn.MSELoss(),
    optimizer_factory=lambda params, lr: torch.optim.AdamW(params, lr=lr),
    # Electrostatic regularization is helpful in higher dimensions
    regularization_func=GaussianDictLayer.electrostatic_regularization,
    regularization_gamma=0.001,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

# Create initial bounds for the N-dimensional space
domain_limits = (-2.0, 2.0)
initial_bounds_list = [[domain_limits[0]] * n_features, [domain_limits[1]] * n_features]
initial_bounds_tensor = torch.tensor(initial_bounds_list, dtype=torch.float32)

# Create T for an N-dimensional grid (e.g., 2 blocks per dimension)
blocks_per_dim = 2
t_list = [blocks_per_dim] * n_features
t_tensor = torch.tensor(t_list)

partition_config = UniformPartitionConfig(
    T=t_tensor,
    initial_bounds=initial_bounds_tensor,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

ssesm_config = SSESMConfig(
    n_features=n_features,
    model_epochs=500,
    sparse_coding_config=sparse_coding_config,
    dict_config=dict_config,
    partition_config=partition_config,
    log_interval=50,
    permutation_times=5,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

experiment = {
    "config": ssesm_config,
    "hyp_set": 'N-dim',
    "n_samples": n_samples,
    "seed": 42,
}

# 4. DATA GENERATION
logger.info(f"Generating dataset for an {n_features}-dimensional paraboloid...")
trainDataset, X_train, y_train, testDataset, X_test, y_test = generate_custom_nd_function_dataset(
    n_samples=n_samples,
    n_dimensions=n_features,
    function=nd_paraboloid,
    function_params={"a": 1.0, "c": 0.0},
    limits=domain_limits,
    mesh_divisions=10  # Note: mesh size grows exponentially with n_features
)
logger.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
logger.info(f"Test data shape: X={X_test.shape}, y={y_test.shape}")

# 5. INSTANTIATE AND TRAIN THE MODEL
model = SSESM(**experiment, logger=logger)

try:
    logger.info("Training N-dimensional SSESM model...")
    model.partial_fit(X_train, y_train)
    
    logger.info("Evaluating model performance...")
    y_predicted, time, mse_value = model.performance_stats(X_test, y_test)
    
    logger.info(
        "--- N-Dimensional Experiment Complete ---"
    )
    logger.info(
        f"Model: {model.__class__.__name__}, "
        f"MSE Value = {mse_value:.6f}, "
        f"Training Time = {time:.2f} minutes"
    )

except KeyboardInterrupt:
    print("\nTraining aborted by user.")
