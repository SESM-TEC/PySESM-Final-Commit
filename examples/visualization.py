"""
Visualization Hooks.

Provides the `VisualizerHook` class for inspecting and rendering the evolution
of the dictionary and sparse codes during model training.

Copyright (c) 2023-2025, Tecnológico de Costa Rica
All rights reserved.

This source code is licensed under the BSD 3-Clause License found in the
LICENSE file in the root directory of this source tree.

SPDX-License-Identifier: BSD-3-Clause
"""

import logging
import torch
import numpy as np
import imageio.v2 as imageio
from pathlib import Path
import datetime

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle

from pysesm.models.SESM import SESM

# Get a logger instance
logger = logging.getLogger(__name__)

class VisualizerHook:
    """
    A hook class to visualize the state of the Gaussian dictionary during
    model training. It's generalized to handle multi-block partitions by
    displaying block boundaries and their specific activations (h) within a single plot.
    
    It saves frames of the visualization and can compile them into a video.
    """
    def __init__(self, model: SESM, ax: plt.Axes, X_train: torch.Tensor,
                 gt_mu: list, gt_sigma: list, 
                 output_dir: str = "animation_frames",
                 plot_limits: tuple = ((-2.5, 2.5), (-2.5, 2.5)),
                 headless: bool = False):
        self.model = model
        self.ax = ax
        self.X_train = X_train.detach().cpu()
        self.ground_truth_mu = [mu.cpu() for mu in gt_mu]
        self.ground_truth_sigma = [s.cpu() for s in gt_sigma]
        self.plot_limits = plot_limits
        self.fig = ax.get_figure()
        self.headless = headless
        
        # Attributes for video frame generation.
        provided_path = Path(output_dir)
        if provided_path.is_dir():
            # New behavior: A pre-created, unique directory was provided.
            self.output_path = provided_path
        else:
            # Retro-compatible behavior: Create a unique sub-directory inside the base path.
            run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_path = provided_path / f"run_{run_timestamp}"

        self.frames_path = self.output_path / "frames"
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
        """
        eigenvalues, eigenvectors = torch.linalg.eigh(Sigma)
        # Clamp small negative eigenvalues that can appear due to numerical instability
        eigenvalues = torch.clamp(eigenvalues, min=0)
        v_max = eigenvectors[:, -1]
        angle = np.degrees(np.arctan2(v_max[1], v_max[0]))
        width, height = 4 * torch.sqrt(eigenvalues) # 2 standard deviations
        ellipse = Ellipse(xy=mu, width=width, height=height, angle=angle,
                          facecolor=color, edgecolor=color, alpha=alpha_fill,
                          linestyle=linestyle, linewidth=1)
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

        # --- Clear and Setup Plot ---
        self.ax.cla()
        self.ax.scatter(self.X_train[:, 0], self.X_train[:, 1], c='gray', 
                        alpha=0.1, s=10, label='Train Data')

        # --- Get Model State ---
        params = info['dictionary_params'].detach().cpu()
        active_blocks = self.model.partition_manager.retrieve_active_blocks()
        if not active_blocks:
            return

        n_features = self.model.n_features
        n_functions = self.model.n_functions
        num_rho_params = n_features * (n_features + 1) // 2
        rho_params = params[:num_rho_params, :]
        mu_params = params[-n_features:, :]
        
        # --- Draw Ground Truth and Block Context ---
        for mu_gt, sigma_gt in zip(self.ground_truth_mu, self.ground_truth_sigma):
            self._draw_ellipse(mu_gt, sigma_gt, 'green', '--', 0.1)

        block_colors = plt.cm.get_cmap('tab10', len(active_blocks))

        # --- Iterate Over Blocks to Visualize Local Activations ---
        for i_block, block in enumerate(active_blocks):
            color = block_colors(i_block)
            block_min_coords = block.block_scope[0].cpu()

            # Retrieve the actual normalization scale used by the block.
            # Fallback to block_size if not set (legacy/unnormalized behavior).
            if block.normalization_scale is not None:
                norm_scale = block.normalization_scale.cpu()
            else:
                norm_scale = block.block_size.cpu()           
            
            # Draw block boundary
            rect = Rectangle(block_min_coords, *block.block_size.cpu(),
                             linewidth=1, edgecolor=color, facecolor='none', 
                             linestyle=':', alpha=0.8)
            self.ax.add_patch(rect)

            h_magnitudes = torch.abs(block.sparse_coding_layer.h).detach().cpu().numpy().flatten()

            # Draw dictionary functions in the context of this block
            for i in range(n_functions):
                mu_norm = mu_params[:, i]
                rho_norm = rho_params[:, i]

                # Denormalize mu and Sigma using this block's scope
                mu_orig = mu_norm * norm_scale + block_min_coords
                
                A = torch.zeros(n_features, n_features)
                indices = torch.triu_indices(n_features, n_features)
                A[indices[0], indices[1]] = rho_norm
                
                G = A.T @ A
                try:
                    Sigma_norm = torch.linalg.inv(G)
                except torch.linalg.LinAlgError:
                    Sigma_norm = torch.eye(n_features) # Fallback

                # Scale covariance matrix properly (handles anisotropic scaling)
                scale_mat = torch.diag(norm_scale)
                Sigma_orig = scale_mat @ Sigma_norm @ scale_mat

                # Draw dictionary ellipse (subtly)
                self._draw_ellipse(mu_orig, Sigma_orig, 'red', '-', 0.02)

                # Draw activation circle for this block
                self.ax.scatter(mu_orig[0], mu_orig[1], s=800 * h_magnitudes[i] + 5, 
                                c=[color], alpha=0.7, edgecolors='black', zorder=10,
                                label=f'Block {block.block_index}' if i == 0 else "")

        # --- Configure and Save Plot ---
        self.ax.set_title(f'Dictionary Evolution (Multi-Block)')
        self.ax.set_xlabel('Feature 1'); self.ax.set_ylabel('Feature 2')
        self.ax.set_xlim(*self.plot_limits[0]); self.ax.set_ylim(*self.plot_limits[1])
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.legend(loc='upper right')

        # Add loss and epoch info text
        losses_text = []
        if 'sparse_coding_losses' in info:
            losses_text.append(f"SC Loss: {float(info['sparse_coding_losses'][-1]):.6f}")
        if 'dictionary_losses' in info:
            losses_text.append(f"Dict Loss: {float(info['dictionary_losses'][-1]):.6f}")
        progress = (epoch + 1) / total_epochs * 100
        losses_text.append(f"Epoch: {epoch + 1}/{total_epochs} ({progress:.1f}%)")
        text_content = "\n".join(losses_text)
        
        self.ax.text(0.02, 0.98, text_content, transform=self.ax.transAxes, fontsize=9,
                     fontfamily='monospace', verticalalignment='top', 
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))
        
        if not self.headless:
            plt.pause(0.01)

        # Save frame

        # --- Proactive size enforcement to avoid rounding errors ---
        # Define target size in pixels and a "clean" DPI to calculate exact inches.
        # This prevents floating point rounding issues in Matplotlib's backend that
        # can cause frames to have slightly different sizes (e.g., 1279 vs 1280 pixels).
        TARGET_DPI = 160
        TARGET_WIDTH_PX = 1280
        TARGET_HEIGHT_PX = 960
        width_inches = TARGET_WIDTH_PX / TARGET_DPI
        height_inches = TARGET_HEIGHT_PX / TARGET_DPI
        
        frame_filename = self.frames_path / f"frame_{self.frame_count:04d}.png"

        self.fig.set_size_inches(width_inches, height_inches)
        self.fig.savefig(frame_filename, dpi=TARGET_DPI, facecolor='white')
        
        self.frame_files.append(frame_filename)
        self.frame_count += 1

    def create_video(self, video_name: str ="dictionary_evolution.mp4", fps=10):
        """Compiles the saved frames into a video and cleans up the files."""
        if not self.frame_files:
            logger.warning("No frames were saved, skipping video creation.")
            return

        video_path = self.output_path / video_name
        stem, suffix = video_path.stem, video_path.suffix
        
        counter = 1
        while video_path.exists():
            video_path = self.output_path / f"{stem}_{counter}{suffix}"        
            counter += 1

        logger.info(f"Creating video '{video_path}' from {len(self.frame_files)} frames...")
        
        with imageio.get_writer(video_path, fps=fps) as writer:
            for filename in self.frame_files:
                image = imageio.imread(str(filename))
                writer.append_data(image)
        
        logger.info(f"Video saved successfully to '{video_path.resolve()}'!")
        
        logger.info("Cleaning up temporary frame files...")
        for filename in self.frame_files:
            filename.unlink()
        try:
            self.frames_path.rmdir()
        except OSError as e:
            logger.warning(f"Could not remove frame directory {self.frames_path}: {e}")
        logger.info("Cleanup complete.")
