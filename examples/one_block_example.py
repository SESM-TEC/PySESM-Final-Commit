'''
Copyright (C) 2023-2025 Tecnológico de Costa Rica

This script provides a trivial example of using the SESM framework 
to represent a dataset of three Gaussian distributions with a single block.

Authors: The SESM Team 
License: 
'''

import logging
import torch
import numpy as np
import imageio.v2 as imageio
from pathlib import Path
import datetime

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from pysesm.models.SESM import SESM
from pysesm.models.SSESM import SSESM, SSESMConfig
from pysesm.models.BSESM import BSESM, BSESMConfig
from pysesm.sparse_coding import ISTAConfig, StepSizeMethod
#from pysesm.sparse_coding.FISTALayer import FISTAConfig, RestartStrategy, MomentumScheme, StepSizeMethod
#from pysesm.sparse_coding import ADMMConfig
from pysesm.dictionaries import GaussianDictConfig, GaussianDictLayer
from pysesm.blocks.UniformPartitionManager import UniformPartitionConfig
from pysesm.utils.loggers import setup_logger
from pysesm.utils_dataset.generate_dataset import generate_gaussian_dataset
from pysesm.utils.plot_and_save_stats import plot_surface
#from pysesm.utils.metric_loggers import *
from pysesm.enums.DeviceTargetEnum import DeviceTarget
#from pysesm.device_manager.DeviceManager import DeviceManager
from mpl_toolkits.mplot3d import Axes3D

# --- Custom Loss Function Wrappers ---

class KLDivLossWrapper(torch.nn.Module):
    """
    A wrapper for KL Divergence Loss that ensures inputs are valid probability
    distributions by ensuring non-negativity and normalizing them.
    """
    def __init__(self, reduction='mean'):
        super(KLDivLossWrapper, self).__init__()
        self.kl_loss = torch.nn.KLDivLoss(reduction=reduction)
        
    def forward(self, inputs, targets):
        # Ensure non-negativity and add a small epsilon for numerical stability.
        inputs = torch.nn.functional.relu(inputs) + 1e-8
        targets = torch.nn.functional.relu(targets) + 1e-8
        
        # Normalize tensors to make them proper distributions.
        inputs_normalized = inputs / torch.sum(inputs)
        targets_normalized = targets / torch.sum(targets)
        
        # Apply log transformation for the KLDivLoss function.
        log_inputs = torch.log(inputs_normalized)
        
        # Calculate KL divergence.
        loss = self.kl_loss(log_inputs, targets_normalized)
        
        return loss

class CrossEntropyLossWrapper(torch.nn.Module):
    """
    Custom Cross-Entropy loss that normalizes both inputs and targets to 
    treat them as probability distributions before calculation.
    """
    def __init__(self, reduction='mean', epsilon=1e-10):
        super(CrossEntropyLossWrapper, self).__init__()
        self.reduction = reduction
        self.epsilon = epsilon
        
    def forward(self, inputs, targets):
        # Ensure non-negativity.
        inputs = torch.nn.functional.relu(inputs) + self.epsilon
        targets = torch.nn.functional.relu(targets) + self.epsilon
        
        # Normalize to make them proper distributions.
        inputs_normalized = inputs / torch.sum(inputs)
        targets_normalized = targets / torch.sum(targets)
        
        # Cross-entropy = -sum(P * log(Q)), where P=targets, Q=inputs.
        cross_entropy = -torch.sum(
            targets_normalized * torch.log(inputs_normalized + self.epsilon)
        )
        
        return cross_entropy

class JensenShannonLossWrapper(torch.nn.Module):
    """
    Custom Jensen-Shannon divergence loss. JS divergence is a symmetrized 
    and smoothed version of the KL divergence.
    JS(P||Q) = 0.5 * (KL(P||M) + KL(Q||M)) where M = 0.5 * (P + Q).
    """
    def __init__(self, reduction='mean', epsilon=1e-10):
        super(JensenShannonLossWrapper, self).__init__()
        self.reduction = reduction
        self.epsilon = epsilon
        
    def forward(self, inputs, targets):
        # Ensure non-negativity.
        inputs = torch.nn.functional.relu(inputs) + self.epsilon
        targets = torch.nn.functional.relu(targets) + self.epsilon
        
        # Normalize to make them proper distributions.
        inputs_normalized = inputs / torch.sum(inputs)
        targets_normalized = targets / torch.sum(targets)
        
        # Compute the average distribution M.
        M = 0.5 * (inputs_normalized + targets_normalized)
        
        # Compute KL(targets || M).
        ratio1 = (targets_normalized + self.epsilon) / (M + self.epsilon)
        kl1 = torch.sum(targets_normalized * torch.log(ratio1))
        
        # Compute KL(inputs || M).
        ratio2 = (inputs_normalized + self.epsilon) / (M + self.epsilon)
        kl2 = torch.sum(inputs_normalized * torch.log(ratio2))
        
        # JS divergence formula.
        js_divergence = 0.5 * (kl1 + kl2)
        
        return js_divergence

# --- Visualization Hook ---

class VisualizerHook:
    """
    A hook class to visualize the state of the Gaussian dictionary during
    model training. It saves frames of the visualization and can compile 
    them into a video.
    """
    def __init__(self, model: SESM, ax: plt.Axes, X_train: torch.Tensor,
                 gt_mu: list, gt_sigma: list, 
                 output_dir: str = "animation_frames"):
        self.model = model
        self.ax = ax
        self.X_train = X_train.detach().cpu()
        self.ground_truth_mu = [mu.cpu() for mu in gt_mu]
        self.ground_truth_sigma = [s.cpu() for s in gt_sigma]
        self.fig = ax.get_figure()
        
        # Attributes for video frame generation.
        run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.frames_path = Path(output_dir) / f"run_{run_timestamp}"
        self.frames_path.mkdir(parents=True, exist_ok=True)
        self.frame_files = []
        self.frame_count = 0
        logger.info(
            f"Animation frames will be saved to: {self.frames_path.resolve()}"
        )

    def _draw_ellipse(self, mu: torch.Tensor, Sigma: torch.Tensor, color: str, 
                      linestyle: str, alpha_fill: float):
        """
        Draws an ellipse on the hook's axes to represent a Gaussian distribution.

        Args:
            mu (torch.Tensor): The mean of the Gaussian (center of the ellipse). 
                               Shape: (2,).
            Sigma (torch.Tensor): The covariance matrix of the Gaussian. 
                                  Used to determine the shape and orientation. 
                                  Shape: (2, 2).
            color (str): The color for the ellipse's edge and fill.
            linestyle (str): The line style for the edge (e.g., '--', '-').
            alpha_fill (float): The alpha transparency for the ellipse's fill.
        """
        eigenvalues, eigenvectors = torch.linalg.eigh(Sigma)
        v_max = eigenvectors[:, -1]
        angle = np.degrees(np.arctan2(v_max[1], v_max[0]))
        width, height = 4 * torch.sqrt(eigenvalues) # 2 standard deviations
        ellipse = Ellipse(xy=mu, width=width, height=height, angle=angle,
                          facecolor=color, edgecolor=color, alpha=alpha_fill,
                          linestyle=linestyle, linewidth=2)
        self.ax.add_patch(ellipse)

    def __call__(self, info: dict):
        """
        This method is called by the SESM model at specified intervals.
        It extracts model parameters and generates a plot frame.
        """
        epoch = info.get('model_epoch', 0)
        log_interval = self.model.config.log_interval
        total_epochs = self.model.config.model_epochs

        # Only render a frame at specified intervals to save time.
        is_log_epoch = (epoch + 1) % log_interval == 0
        is_last_epoch = epoch == total_epochs - 1
        if not (epoch == 0 or is_log_epoch or is_last_epoch):
            return

        params = info['dictionary_params'].detach().cpu()
        
        active_block = self.model.partition_manager.retrieve_active_blocks()[0]
        default_h = [active_block.sparse_coding_layer.h]
        h_per_block = info.get('h_per_block', default_h)

        h_magnitudes = torch.abs(h_per_block[0]).detach().cpu().numpy().flatten()
        
        n_features = self.model.n_features
        n_functions = self.model.n_functions
        num_rho_params = n_features * (n_features + 1) // 2
        rho_params = params[:num_rho_params, :]
        mu_params = params[-n_features:, :]

        # Clear and redraw the plot for the current state.
        self.ax.cla()
        self.ax.scatter(self.X_train[:, 0], self.X_train[:, 1], c='gray', 
                        alpha=0.1, s=10, label='Train Data')
        
        # Draw ground truth ellipses.
        for mu_gt, sigma_gt in zip(self.ground_truth_mu, self.ground_truth_sigma):
            self._draw_ellipse(mu_gt, sigma_gt, 'green', '--', 0.1)

        # Draw learned dictionary function ellipses.
        for i in range(n_functions):
            mu = mu_params[:, i]
            rho = rho_params[:, i]
            A = torch.zeros(n_features, n_features)
            indices = torch.triu_indices(n_features, n_features)
            A[indices[0], indices[1]] = rho
            G = A.T @ A
            try:
                Sigma = torch.linalg.inv(G)
            except torch.linalg.LinAlgError:
                Sigma = torch.eye(n_features) # Fallback for singular matrix
            self._draw_ellipse(mu, Sigma, 'red', '-', 0.05)
            # Scatter plot for means, size proportional to activation magnitude.
            self.ax.scatter(mu[0], mu[1], s=800 * h_magnitudes[i] + 5, 
                            c='red', alpha=0.7, edgecolors='black', zorder=10)

        # Configure plot aesthetics.
        self.ax.set_title(f'Dictionary Evolution')
        self.ax.set_xlabel('Feature 1'); self.ax.set_ylabel('Feature 2')
        self.ax.set_xlim(-2.5, 2.5); self.ax.set_ylim(-2.5, 2.5)
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.ax.set_aspect('equal', adjustable='box')

        # --- Display Loss and Epoch Information ---
        losses_text = []
        
        # Extract sparse coding loss.
        if 'sparse_coding_losses' in info:
            sparse_losses = info['sparse_coding_losses']
            if hasattr(sparse_losses, '__len__') and len(sparse_losses) > 0:
                sparse_loss = sparse_losses[-1]
            else:
                sparse_loss = sparse_losses
            losses_text.append(f"Sparse Loss: {float(sparse_loss):.6f}")

        # Extract dictionary loss.
        if 'dictionary_losses' in info:
            dict_losses = info['dictionary_losses']
            if hasattr(dict_losses, '__len__') and len(dict_losses) > 0:
                dict_loss = dict_losses[-1]
            else:
                dict_loss = dict_losses
            losses_text.append(f"Dict Loss: {float(dict_loss):.6f}")

        progress = (epoch + 1) / total_epochs * 100
        losses_text.append(f"Epoch: {epoch + 1}/{total_epochs} ({progress:.1f}%)")
        
        text_content = "\n".join(losses_text)

        # Add text box to the plot.
        bbox_props = dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                          alpha=0.9, edgecolor='navy')
        self.ax.text(0.02, 0.98, text_content, 
                     transform=self.ax.transAxes,
                     fontsize=9,
                     fontfamily='monospace',
                     verticalalignment='top',
                     bbox=bbox_props)
        
        plt.pause(0.01)

        # Save the current figure as a frame.
        frame_filename = self.frames_path / f"frame_{self.frame_count:04d}.png"
        
        # Use a fixed size and DPI for consistent video frames.
        target_width, target_height = 1280, 960
        dpi = 150
        self.fig.set_size_inches(target_width / dpi, target_height / dpi)
        self.fig.savefig(frame_filename, dpi=dpi, facecolor='white')
    
        self.frame_files.append(frame_filename)
        self.frame_count += 1

    def create_video(self, video_name="dictionary_evolution.mp4", fps=10):
        """Compiles the saved frames into a video and cleans up the files."""
        if not self.frame_files:
            logger.warning("No frames were saved, skipping video creation.")
            return

        # Find an available filename to avoid overwriting.
        output_path = Path(video_name)
        stem, suffix = output_path.stem, output_path.suffix
        counter = 1
        while output_path.exists():
            output_path = Path(f"{stem}_{counter}{suffix}")
            counter += 1

        logger.info(
            f"Creating video '{output_path}' from {len(self.frame_files)} frames..."
        )
        
        with imageio.get_writer(output_path, fps=fps) as writer:
            for filename in self.frame_files:
                image = imageio.imread(str(filename))
                writer.append_data(image)
        
        logger.info(f"Video saved successfully to '{output_path.resolve()}'!")
        
        # Cleanup temporary frame files and directory.
        logger.info("Cleaning up temporary frame files...")
        for filename in self.frame_files:
            filename.unlink()
        self.frames_path.rmdir()
        logger.info("Cleanup complete.")
    
# --- Main Script ---

# 1. SETUP LOGGER
logger = setup_logger(level=logging.DEBUG)

# 2. DEFINE MODEL CONFIGURATIONS
n_functions = 100
n_features = 2


sparse_coding_config = ISTAConfig(
    epochs=100,
    alpha=0.15,
    lambd=0.001,
    step_size_method=StepSizeMethod.FROBENIUS,  # POWER_ITERATION,
    power_iterations=10,
    n_functions=n_functions,
    criterion=torch.nn.MSELoss()
)
# sparse_coding_config = FISTAConfig(
#     epochs=400,
#     alpha = 0.020,
#     lambd = 0.00001,
#     step_size_method = StepSizeMethod.FROBENIUS,  # POWER_ITERATION,
#     power_iterations = 10,
#     early_stopping = False,
#     n_functions = n_functions,
#     restart_strategy = RestartStrategy.ADAPTIVE, # .NONE,
#     momentum_scheme = MomentumScheme.MONOTONIC, # .ORIGINAL,
#     criterion = torch.nn.MSELoss(),
# )
# sparse_coding_config = ADMMConfig(
#     epochs = 100,
#     rho = 0.1,            # Penalty parameter
#     alpha = 1.5,          # Relaxation parameter (>1.0 for over-relaxation)
#     lambda_scaling = 1.0, # Lambda scaling factor
#     lambd = 0.00001,      # L1 regularization strength
#     abs_tol = 1e-4,       # Absolute tolerance
#     rel_tol = 1e-2,       # Relative tolerance
#     n_functions = n_functions,
#     criterion = torch.nn.MSELoss()
# )

dict_config = GaussianDictConfig(
    epochs = 50,
    alpha = 0.01,
    # criterion = torch.nn.MSELoss(),
    # criterion = KLDivLossWrapper(),
    criterion = JensenShannonLossWrapper(),
    optimizer_factory = lambda params, lr: torch.optim.SGD(
        params, lr=lr, momentum=0.1
    ),
    mu_epochs = 10,
    rho_epochs = 10,
    split_mu_rho = False,
    eig_range = [0.05, 0.2],
    mu_range = [-2.0, 2.0],
    #regularization_func = None,
    regularization_func = GaussianDictLayer.electrostatic_regularization, 
    #regularization_func = GaussianDictLayer.gram_regularization, # or None
    regularization_gamma = 0.005
)

partition_config = UniformPartitionConfig(
    T=1,
    initial_bounds = torch.tensor([[-2, -2], [2, 2]], dtype=torch.float32),
    activity_threshold=0,
    overlap_ratio=0.25
)
# partition_config = AdaptativePartitionConfig(
#         maxNodeSize=251,
#         maxSplitsBeforeRestart=5,
#         overlap_ratio=0.1)


ssesm_config = SSESMConfig(
    n_features = n_features,
    model_epochs = 5000,
    sparse_coding_config = sparse_coding_config,
    dict_config = dict_config,
    partition_config = partition_config,
    log_interval=25,
    permutation_times=1
)

bsesm_config = BSESMConfig(
    n_features = n_features,
    model_epochs = 7500,
    sparse_coding_config = sparse_coding_config,
    dict_config = dict_config,
    partition_config = partition_config,
    log_interval=100,
)

# 3. DEFINE EXPERIMENT PARAMETERS
which_sesm="ssesm" # "ssesm" or "bsesm"

experiment = {
    "config": bsesm_config if which_sesm=="bsesm" else ssesm_config,
    "hyp_set": 1,
    "n_samples": 500,
    "seed": 45,
    "iter": 0,
    "device_map": {
        DeviceTarget.GLOBAL: "cpu",
        DeviceTarget.SPARSE_CODING_LAYER: "cpu",
        DeviceTarget.DICTIONARY_LAYER: "cuda",
        DeviceTarget.PARTITION_MANAGER: "cpu"
    },
    
    #"dict_layer_hook": lambda info: log_to_WB("DictLayer", info, logger=logger, project_name="sesm-test"),
    #"ista_layer_hook": lambda info: log_to_WB("IstaLayer", info, logger=logger, project_name="sesm-test"),
    #"dict_layer_hook": lambda info: log_to_console("DictLayer", info),
    #"ista_layer_hook": lambda info: log_to_console("IstaLayer", info),   
    #"sesm_hook": lambda info: log_to_WB("SESM", info, logger=logger, project_name="sesm-test")
}

def show_all_h(model: SESM, logger: logging.Logger, threshold: float = 1e-6):
    """
    Inspects and logs the activation vectors (h) for all active blocks.

    Args:
        model (SESM): The trained SESM model instance.
        logger (logging.Logger): The logger instance for output.
        threshold (float): Threshold to consider a component of h as non-zero.
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
            logger.info(f"    Amplitude: {block.amplitude}")
            logger.info(f"    h-vector (shape {h_tensor.shape}):")
            logger.info(f"{h_tensor.numpy().flatten()}")
            logger.info(f"    Non-zero components: {non_zero} / {total}")
            logger.info(f"    Sparsity: {sparsity:.2f}%")
            logger.info(f"    L1 Norm of h: {torch.norm(h_tensor, p=1):.4f}")
            logger.info(f"    L2 Norm of h: {torch.norm(h_tensor, p=2):.4f}")
        else:
            logger.warning(
                f"  Block {block_index_str}: Sparse coding layer or "
                f"h-vector not found."
            )
    logger.info("--- END OF H-VECTOR INSPECTION ---\n")

# 4. DATA GENERATION
(trainDataset, X_train, y_train,
 testDataset, X_test, y_test,
 gt_mu, gt_sigma) = generate_gaussian_dataset(
    n_samples=experiment["n_samples"]
)

# 5. SETUP VISUALIZATION AND MODEL
fig_hook, ax_hook = plt.subplots(figsize=(10, 8))
plt.ion() # Enable interactive mode for plotting.

if which_sesm == "bsesm":
    model = BSESM(**experiment, logger=logger)
else:
    model = SSESM(**experiment, logger=logger)

# Create and install the visualization hook AFTER the model is instantiated.
visual_hook = VisualizerHook(model, ax_hook, X_train, gt_mu, gt_sigma)
model.sesm_hook = visual_hook
    
# 6. TRAIN AND EVALUATE THE MODEL
try:
    logging.info("Training model %s", model.__class__.__name__)
    model.partial_fit(X_train, y_train)
    
    if which_sesm == "ssesm":
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
        visual_hook.create_video()

print("\nDisplaying final plots. Close all plot windows to exit.")
plt.show(block=True)

