"""
Video Demo Script.

Runs a training session and uses the visualization hook to generate an
MP4 video of the dictionary evolution, saving it to the documentation folder.

Copyright (c) 2023-2025, Tecnológico de Costa Rica
All rights reserved.

This source code is licensed under the BSD 3-Clause License found in the
LICENSE file in the root directory of this source tree.

SPDX-License-Identifier: BSD-3-Clause
"""

import os
import logging
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from pysesm.utils.loggers import setup_logger
from pysesm.models.SSESM import SSESM, SSESMConfig
from pysesm.blocks import UniformPartitionConfig
from pysesm.dictionaries import GaussianDictConfig, GaussianDictLayer
from pysesm.sparse_coding import ISTAConfig, StepSizeMethod

from pysesm.utils_dataset.generate_dataset import generate_gaussian_dataset
from pysesm.utils_dataset.gaussian_covariance_density import generate_nondiag_covariance_matrices

# NOTE: This script lives in the same folder as `visualization.py`.
# That allows a simple relative import like below.
from visualization import VisualizerHook


def main():
    logger = setup_logger(level=logging.INFO)

    # --- Overall Model Parameters ---
    n_features = 2
    n_functions = 100

    # 1) Partition: single block over [-2,2] x [-2,2]
    partition_config = UniformPartitionConfig(
        T=1,
        initial_bounds=torch.tensor([[-2, -2], [2, 2]], dtype=torch.float32),
    )

    # 2) Dictionary: Gaussian functions
    dict_config = GaussianDictConfig(
        epochs=100,
        alpha=0.01,
        criterion=torch.nn.MSELoss(),
        optimizer_factory=lambda params, lr: torch.optim.AdamW(params, lr=lr),
        regularization_func=GaussianDictLayer.electrostatic_regularization,
        regularization_gamma=0.001,
    )

    # 3) Sparse Coding: ISTA
    sparse_coding_config = ISTAConfig(
        epochs=150,
        alpha=0.15,
        lambd=0.005,
        step_size_method=StepSizeMethod.FROBENIUS,
        n_functions=n_functions,
        criterion=torch.nn.MSELoss(),
    )

    # 4) SSESM Config
    ssesm_config = SSESMConfig(
        n_features=n_features,
        model_epochs=500,
        partition_config=partition_config,
        dict_config=dict_config,
        sparse_coding_config=sparse_coding_config,
        log_interval=20,
    )

    # --- Dataset: 3 non-diagonal Gaussians
    sigma1, sigma2, sigma3 = generate_nondiag_covariance_matrices()
    (
        trainDataset,
        X_train,
        y_train,
        testDataset,
        X_test,
        y_test,
        gt_mu,
        gt_sigma,
    ) = generate_gaussian_dataset(n_samples=500, variances=[sigma1, sigma2, sigma3])

    # --- Model
    experiment = {
        "config": ssesm_config,
        "seed": 45,
    }
    model = SSESM(**experiment, logger=logger)

    # --- Visualization hook
    fig_hook, ax_hook = plt.subplots(figsize=(10, 8))
    plt.ion()
    visual_hook = VisualizerHook(
        model,
        ax_hook,
        X_train,
        gt_mu,
        gt_sigma,
        plot_limits=((-5, 5), (-5, 5)),
    )
    model.sesm_hook = visual_hook

    # --- Train and produce video
    logging.info("Training model with visualization hook...")
    try:
        model.partial_fit(X_train, y_train)
    finally:
        # Save the video under docs/pics so it can be embedded by the manual
        repo_root = Path(__file__).resolve().parents[1]
        out_dir = repo_root / "docs" / "pics"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "training_evolution.mp4"
        visual_hook.create_video(video_name=str(out_path))
        logging.info(f"Video saved to: {out_path}")

    # Optional: keep the window open if running interactively
    plt.show(block=True)


if __name__ == "__main__":
    main()
