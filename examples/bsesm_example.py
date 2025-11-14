'''
Copyright (C) 2023-2025 Tecnológico de Costa Rica

This script provides a trivial example of using the SESM framework 
to represent a dataset of three Gaussian distributions with a single block.

Authors: The SESM Team 
License: 
'''

import logging
import torch
import argparse
import json
import dataclasses
from datetime import datetime
from pathlib import Path

# --- Argument Parsing & Headless Setup ---
# This must happen BEFORE importing matplotlib.pyplot
parser = argparse.ArgumentParser(description="Run a BSESM experiment.")
parser.add_argument('--headless', action='store_true', 
                    help="Run in headless mode (no plots displayed, saves files).")
args = parser.parse_args()

if args.headless:
    import matplotlib
    matplotlib.use('Agg')


import matplotlib.pyplot as plt
   
from pysesm.models.SESM import SESM
from pysesm.models.SSESM import SSESM, SSESMConfig
from pysesm.models.BSESM import BSESM, BSESMConfig
from pysesm.sparse_coding import ISTAConfig, StepSizeMethod
from pysesm.sparse_coding.FISTALayer import FISTAConfig, RestartStrategy, MomentumScheme
from pysesm.sparse_coding import ADMMConfig
from pysesm.dictionaries import GaussianDictConfig, GaussianDictLayer
from pysesm.blocks.UniformPartitionManager import UniformPartitionConfig
from pysesm.blocks.AdaptativePartitionManager import AdaptativePartitionConfig
from pysesm.utils.loggers import setup_logger
from pysesm.utils_dataset.generate_dataset import generate_gaussian_dataset
from pysesm.utils_dataset.gaussian_covariance_density import generate_nondiag_covariance_matrices
from pysesm.utils.plot_and_save_stats import plot_surface
from visualization import VisualizerHook # Import from the new centralized file
from mpl_toolkits.mplot3d import Axes3D


# --- Directory and Config Setup ---
run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path(f"results/bsesm_run_{run_timestamp}")
output_dir.mkdir(parents=True, exist_ok=True)

class EnhancedJSONEncoder(json.JSONEncoder):
    """
    A custom JSON encoder to handle non-serializable types like dataclasses,
    lambda functions, and torch.device.
    """
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        if isinstance(o, torch.device):
            return str(o)
        if callable(o):
            if hasattr(o, '__name__'):
                return f"<function {o.__name__}>"
            # Handle callable class instances like torch.nn.MSELoss()
            return f"<instance of {o.__class__.__name__}>"
        try:
            return super().default(o)
        except TypeError:
            return str(o)

def save_config_to_json(config_dict, path):
    """Saves the experiment configuration dictionary to a JSON file."""
    with open(path, 'w') as f:
        json.dump(config_dict, f, cls=EnhancedJSONEncoder, indent=4)



# --- Main Script ---

# 1. SETUP LOGGER
logger = setup_logger(level=logging.INFO)

# 2. DEFINE MODEL CONFIGURATIONS
n_functions = 80
n_features = 2

# --- Partition Manager Configurations (choose one) ---
partition_config = UniformPartitionConfig(
    T=3,
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


# --- Dictionary Configuration ---
dict_config = GaussianDictConfig(
    epochs=58,
    alpha=0.0002525,
    criterion=torch.nn.MSELoss(),
    optimizer_factory=lambda params, lr: torch.optim.AdamW(
        params, lr=lr, weight_decay=0.00873
    ),
    # optimizer_factory = lambda params, lr: torch.optim.SGD(
    #    params, lr=lr, momentum=0.1
    # ),
    mu_epochs=10,
    rho_epochs=10,
    split_mu_rho=False,
    # These ranges operate in the NORMALIZED space [0, 1] of the block.        
    eig_range=[0.05, 0.2],
    mu_range=[0.0, 1.0],
    regularization_func=GaussianDictLayer.electrostatic_regularization,
    regularization_gamma=0.00001018,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)


# --- Sparse Coding Configurations (choose one) ---
sparse_coding_config = ISTAConfig(
    epochs=235,
    alpha=0.15,
    lambd=0.0009157,
    step_size_method=StepSizeMethod.FROBENIUS,
    power_iterations=10,
    n_functions=n_functions,
    criterion=torch.nn.MSELoss(),
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)
# sparse_coding_config = FISTAConfig(
#     epochs=100,
#     alpha=0.15,
#     lambd=0.005,
#     step_size_method=StepSizeMethod.FROBENIUS,
#     power_iterations=10,
#     n_functions=n_functions,
#     criterion=torch.nn.MSELoss(),
#     restart_strategy=RestartStrategy.ADAPTIVE,
#     momentum_scheme=MomentumScheme.MONOTONIC,
#     device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# )
# sparse_coding_config = ADMMConfig(
#     epochs=100,
#     rho=1.0,
#     alpha=1.5,
#     lambda_scaling=1.0,
#     lambd=0.00001,
#     abs_tol=1e-4,
#     rel_tol=1e-2,
#     n_functions=n_functions,
#     criterion=torch.nn.MSELoss(),
#     device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# )



# --- Main SESM Model Configurations (choose one) ---
bsesm_config = BSESMConfig(
    n_features=n_features,
    model_epochs=640,
    partition_config=partition_config,
    dict_config=dict_config,
    sparse_coding_config=sparse_coding_config,
    log_interval=10,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

# 3. DEFINE EXPERIMENT PARAMETERS
which_sesm = "bsesm" # "ssesm" or "bsesm"

experiment = {
    "config": bsesm_config, # bsesm_config if which_sesm=="bsesm" else ssesm_config,
    "hyp_set": 1,
    "n_samples": 500,
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

save_config_to_json(experiment, output_dir / "config.json")

model = BSESM(**experiment, logger=logger)

# Create and install the visualization hook using the imported, general-purpose class
visual_hook = VisualizerHook(model, ax_hook, X_train, gt_mu, gt_sigma, plot_limits=((-5, 5), (-5, 5)),
                             output_dir=output_dir, headless=args.headless)
model.sesm_hook = visual_hook
    
# 6. TRAIN AND EVALUATE THE MODEL
try:
    logging.info("Training model %s", model.__class__.__name__)
    model.partial_fit(X_train, y_train)
    
    show_all_h(model, logger)
        
    y_predicted, time, mse_value = model.performance_stats(X_test, y_test)
    logging.info(
        "Model: %s, MSE Value = %.6f, time = %.6f",
        model.__class__.__name__, mse_value, time
    )
    
    final_fig = plot_surface(test_dataset=testDataset,
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
        visual_hook.create_video(video_name="bsesm_evolution.mp4")

if args.headless:
    final_figure_path = output_dir / "final_surface_plot.png"
    final_fig.savefig(final_figure_path, dpi=160)
    logger.info(f"Final plot saved to {final_figure_path}")
    print(f"\nHeadless run finished. Results saved in: {output_dir.resolve()}")
else:
    print("\nDisplaying final plots. Close all plot windows to exit.")
    plt.show(block=True)
