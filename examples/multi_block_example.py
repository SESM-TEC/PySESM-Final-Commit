"""
Multi-Block Example.

Demonstrates the SESM framework representing a dataset of three Gaussian
distributions using a multi-block partition (e.g., a 2x2 uniform grid).

Copyright (c) 2023-2025, Tecnológico de Costa Rica
All rights reserved.

This source code is licensed under the BSD 3-Clause License found in the
LICENSE file in the root directory of this source tree.

SPDX-License-Identifier: BSD-3-Clause
"""

import logging
import torch
import matplotlib.pyplot as plt

from pysesm.models.SESM import SESM
from pysesm.models.SSESM import SSESM, SSESMConfig
from pysesm.sparse_coding import ISTAConfig, StepSizeMethod
from pysesm.dictionaries import GaussianDictConfig, GaussianDictLayer
from pysesm.blocks.UniformPartitionManager import UniformPartitionConfig
from pysesm.utils.loggers import setup_logger
from pysesm.utils_dataset.generate_dataset import generate_gaussian_dataset
from pysesm.utils_dataset.gaussian_covariance_density import generate_nondiag_covariance_matrices
from pysesm.utils.plot_and_save_stats import plot_surface
from visualization import VisualizerHook  # Import from the new centralized file
from mpl_toolkits.mplot3d import Axes3D

# --- Main Script ---

# 1. SETUP LOGGER
logger = setup_logger(level=logging.INFO)

# 2. DEFINE MODEL CONFIGURATIONS
n_functions = 20  
n_features = 2

partition_config = UniformPartitionConfig(
    T=2,  
    initial_bounds=torch.tensor([[-2, -2], [2, 2]], dtype=torch.float32),
    activity_threshold=0,
    overlap_ratio=0.25,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

# partition_config = AdaptativePartitionConfig(
#     maxNodeSize=251,
#     maxSplitsBeforeRestart=5,
#     overlap_ratio=0.1,
#     device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# )



dict_config = GaussianDictConfig(
    epochs=25,
    alpha=0.01,
    criterion=torch.nn.MSELoss(),
    optimizer_factory=lambda params, lr: torch.optim.AdamW(
        params, lr=lr, weight_decay=0.001 # Added small weight decay
    ),
    mu_epochs=10,
    rho_epochs=10,
    split_mu_rho=False,
    # These ranges operate in the NORMALIZED space [0, 1] of the block.
    eig_range=[0.05, 0.2],
    mu_range=[0.0, 1.0],
    regularization_func=GaussianDictLayer.electrostatic_regularization,
    regularization_gamma=0.001,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

sparse_coding_config = ISTAConfig(
    epochs=50,
    alpha=0.15,
    lambd=0.005,
    step_size_method=StepSizeMethod.FROBENIUS,
    power_iterations=10,
    n_functions=n_functions,
    criterion=torch.nn.MSELoss(),
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

ssesm_config = SSESMConfig(
    n_features=n_features,
    model_epochs=100,  # Increased epochs for the more complex multi-block problem
    partition_config=partition_config,
    dict_config=dict_config,
    sparse_coding_config=sparse_coding_config,
    log_interval=25,
    permutation_times=10, # Permuting blocks is important in SSESM
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

# 3. DEFINE EXPERIMENT PARAMETERS
experiment = {
    "config": ssesm_config,
    "hyp_set": 2, # Changed to distinguish from one-block
    "n_samples": 1000, # More samples for a more complex setup
    "seed": 45,
    "iter": 0
}
    
def show_all_h(model: SESM, logger: logging.Logger, threshold: float = 1e-6):
    """
    Inspects and logs the activation vectors (h) for all active blocks.
    """
    logger.info("\n--- INSPECTING H-VECTORS PER BLOCK ---")
    active_blocks = model.partition_manager.retrieve_active_blocks()
    
    if not active_blocks:
        logger.info("No active blocks found in the model.")
        return

    for block in active_blocks:
        block_index_str = str(block.block_index)
        
        if block.sparse_coding_layer and block.sparse_coding_layer.h is not None:
            h_tensor = block.sparse_coding_layer.h.detach().cpu()
            
            non_zero = torch.sum(torch.abs(h_tensor) > threshold).item()
            total = h_tensor.numel()
            sparsity = (total - non_zero) / total * 100
            
            logger.info(f"  Block {block_index_str}:")
            logger.info(f"    Sparsity: {sparsity:.2f}% ({non_zero}/{total} non-zero)")
            logger.info(f"    L1 Norm of h: {torch.norm(h_tensor, p=1):.4f}")
        else:
            logger.warning(f"  Block {block_index_str}: Sparse coding layer or h-vector not found.")
    logger.info("--- END OF H-VECTOR INSPECTION ---\n")

# 4. DATA GENERATION
# Generate non-diagonal covariance matrices for a more challenging problem
sigma1, sigma2, sigma3 = generate_nondiag_covariance_matrices()
non_diag_sigmas = [sigma1, sigma2, sigma3]

(trainDataset, X_train, y_train,
 testDataset, X_test, y_test,
 gt_mu, gt_sigma) = generate_gaussian_dataset(
    n_samples=experiment["n_samples"],
    variances=non_diag_sigmas  
)

# 5. SETUP VISUALIZATION AND MODEL
fig_hook, ax_hook = plt.subplots(figsize=(10, 8))
plt.ion()

model = SSESM(**experiment, logger=logger)

# Create and install the visualization hook using the imported, general-purpose class
visual_hook = VisualizerHook(model, ax_hook, X_train, gt_mu, gt_sigma, plot_limits=((-5, 5), (-5, 5)))
model.sesm_hook = visual_hook
    
# 6. TRAIN AND EVALUATE THE MODEL
try:
    logging.info("Training model %s with T=%d", model.__class__.__name__, partition_config.T)
    model.partial_fit(X_train, y_train)
    
    show_all_h(model, logger)
        
    y_predicted, time, mse_value = model.performance_stats(X_test, y_test)
    logging.info(
        "Model: %s, MSE Value = %.6f, time = %.6f",
        model.__class__.__name__, mse_value, time
    )

    plot_surface(test_dataset=testDataset,
                 X_train=X_train,
                 y_train=y_train,
                 y_pred=y_predicted,
                 model=model,
                 hypset=experiment["hyp_set"])
    
except KeyboardInterrupt:
    print("\nTraining aborted by user. Creating video from captured frames...")
    
finally:
    # Ensure the video is created even if training is interrupted.
    if 'visual_hook' in locals() and visual_hook is not None:
        visual_hook.create_video(video_name="multi_block_evolution.mp4")

print("\nDisplaying final plots. Close all plot windows to exit.")
plt.show(block=True)
