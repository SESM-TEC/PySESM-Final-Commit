'''
Copyright (C) 2023-2025 Tecnológico de Costa Rica

This script provides a minimal example of using SESM to represent a 
single Gaussian distribution with one dictionary word (n_functions=1) in a single block.
It's an excellent test case for debugging dictionary learning.

Authors: The SESM Team 
License: 
'''

import logging
import torch
import matplotlib.pyplot as plt

from pysesm.models.SESM import SESM
from pysesm.models.SSESM import SSESM, SSESMConfig
from pysesm.sparse_coding import ISTAConfig, StepSizeMethod
from pysesm.dictionaries import GaussianDictConfig
from pysesm.blocks.UniformPartitionManager import UniformPartitionConfig
from pysesm.utils.loggers import setup_logger
from pysesm.utils_dataset.generate_dataset import generate_gaussian_dataset
from pysesm.utils.plot_and_save_stats import plot_surface
from visualization import VisualizerHook
from mpl_toolkits.mplot3d import Axes3D

class JensenShannonLossWrapper(torch.nn.Module):
    """
    Custom Jensen-Shannon divergence loss. JS divergence is a symmetrized 
    and smoothed version of the KL divergence, which can provide better
    gradients when fitting distributions.
    JS(P||Q) = 0.5 * (KL(P||M) + KL(Q||M)) where M = 0.5 * (P + Q).
    """
    def __init__(self, reduction='mean', epsilon=1e-10):
        super(JensenShannonLossWrapper, self).__init__()
        self.reduction = reduction
        self.epsilon = epsilon
        
    def forward(self, inputs, targets):
        # Shift both tensors to ensure non-negativity, mapping the target's minimum to 0.
        min_val = torch.min(targets)
        shifted_inputs = inputs - min_val
        shifted_targets = targets - min_val

        # Apply relu as a safeguard and add epsilon for numerical stability.
        inputs = torch.nn.functional.relu(shifted_inputs) + self.epsilon
        targets = torch.nn.functional.relu(shifted_targets) + self.epsilon
        
        # Normalize to make them proper distributions.
        inputs_normalized = inputs / torch.sum(inputs)
        targets_normalized = targets / torch.sum(targets)
        
        M = 0.5 * (inputs_normalized + targets_normalized)
        
        kl1 = torch.sum(targets_normalized * torch.log((targets_normalized + self.epsilon) / (M + self.epsilon)))
        kl2 = torch.sum(inputs_normalized * torch.log((inputs_normalized + self.epsilon) / (M + self.epsilon)))
        
        return 0.5 * (kl1 + kl2)


# --- Main Script ---

# 1. SETUP LOGGER
logger = setup_logger(level=logging.INFO)

# 2. DEFINE MODEL CONFIGURATIONS
n_functions = 1
n_features = 2

partition_config = UniformPartitionConfig(
    T=1,
    initial_bounds=torch.tensor([[-2, -2], [2, 2]], dtype=torch.float32),
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)


dict_config = GaussianDictConfig(
    epochs=100,
    alpha=0.005,
    #criterion=torch.nn.MSELoss(),
    criterion=JensenShannonLossWrapper(),
    optimizer_factory=lambda params, lr: torch.optim.AdamW(params, lr=lr),
    # No regularization needed for a single dictionary element
    regularization_func=None,
    regularization_gamma=0.0,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

sparse_coding_config = ISTAConfig(
    epochs=0,  # Less epochs needed as h will be trivial (mostly [1])
    alpha=0.1,
    lambd=0.001,  # No sparsity needed for a single function
    step_size_method=StepSizeMethod.FROBENIUS,
    n_functions=n_functions,
    criterion=torch.nn.MSELoss(),
    initial_h=torch.ones(n_functions,1),
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

ssesm_config = SSESMConfig(
    n_features=n_features,
    model_epochs=500,
    sparse_coding_config=sparse_coding_config,
    dict_config=dict_config,
    partition_config=partition_config,
    log_interval=10,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

# 3. DEFINE EXPERIMENT PARAMETERS
experiment = {
    "config": ssesm_config,
    "hyp_set": 0,
    "n_samples": 500,
    "seed": 42,
}
    
# 4. DATA GENERATION
(trainDataset, X_train, y_train,
 testDataset, X_test, y_test,
 gt_mu, gt_sigma) = generate_gaussian_dataset(
    n_samples=experiment["n_samples"],
    means=[(0.5, 0.5)],      # Single target Gaussian
    variances=[0.2],
    weights=[1.0]
)

# 5. SETUP VISUALIZATION AND MODEL
fig_hook, ax_hook = plt.subplots(figsize=(10, 8))
plt.ion()

model = SSESM(**experiment, logger=logger)

# Create and install the visualization hook
visual_hook = VisualizerHook(model, ax_hook, X_train, gt_mu, gt_sigma, plot_limits=((-2.5, 2.5), (-2.5, 2.5)))
model.sesm_hook = visual_hook
    
# 6. TRAIN AND EVALUATE THE MODEL
try:
    logging.info("Training model to fit a single Gaussian...")
    model.partial_fit(X_train, y_train)
    
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
    if 'visual_hook' in locals() and visual_hook is not None:
        visual_hook.create_video(video_name="one_gaussian_evolution.mp4")

print("\nDisplaying final plots. Close all plot windows to exit.")
plt.show(block=True)
