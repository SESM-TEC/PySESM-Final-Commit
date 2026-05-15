"""
Analyze excessive GPU memory allocation behavior.
"""

import copy
import logging
import os
import time

import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from sklearn.metrics import mean_absolute_error, mean_squared_error

import fun

from pysesm.blocks.AdaptivePartitionManager import AdaptivePartitionConfig
from pysesm.blocks.KDTreeStrategy import KDTreeStrategy, KDTreeStrategyConfig
from pysesm.blocks.SESMData import SESMData
from pysesm.blocks.UniformPartitionManager import UniformPartitionConfig
from pysesm.dictionaries import GaussianDictConfig, GaussianDictLayer
from pysesm.models.BSESM import BSESM, BSESMConfig, BSESMSolverStrategy
from pysesm.sparse_coding import ISTAConfig, StepSizeMethod
from pysesm.utils_dataset.generate_dataset import (
    generate_custom_nd_function_dataset,
)

from utils import (
    GPURAMStepSampler,
    plot_multi_method_comparison,
    quadratic_steps,
)


def adamw_factory(params, lr):
    return torch.optim.AdamW(params, lr=lr)


def train_stream_experiment(cfg, logger, func_obj):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gpu_sampler = GPURAMStepSampler(
        enabled=True,
        sample_interval_sec=0.5,
        logger=logger,
        device_index=torch.cuda.current_device() if torch.cuda.is_available() else 0,
    )

    steps = sorted(set(quadratic_steps(cfg.dim, n_steps=cfg.stream_steps)))
    max_samples = max(steps)

    logger.info("Stream Steps for Dim %d: %s", cfg.dim, steps)

    base_path = os.getcwd()

    for run_idx in range(cfg.n_runs):

        torch.manual_seed(cfg.seed + run_idx)
        np.random.seed(cfg.seed + run_idx)

        logger.info(
            "=== RUN %d/%d | Dim: %d | Dataset: %s ===",
            run_idx + 1,
            cfg.n_runs,
            cfg.dim,
            cfg.dataset.name,
        )

        predictions_cache = {}

        dataset_config = {
            "n_samples": max_samples,
            "n_dimensions": cfg.dim,
            "function": func_obj,
            "limits": cfg.dataset.limits,
            "test_random_samples": max_samples*4
        }

        train_data, _, _, test_data, _, _ = (
            generate_custom_nd_function_dataset(**dataset_config)
        )

        x_full_train = train_data["X"].to(device)
        y_full_train = train_data["Z"].to(device)

        x_test = test_data["X"].to(device)
        y_test = test_data["Z"].to(device)

        for method_name in cfg.methods_to_test:

            logger.info(">>> Method: %s", method_name.upper())

            method_conf_path = os.path.join(
                base_path,
                "conf",
                "method",
                f"{method_name}.yaml",
            )

            if os.path.exists(method_conf_path):
                cfg.method = OmegaConf.load(method_conf_path)

            n_atoms = cfg.bsesm_params.atoms_per_dim * cfg.dim

            dict_conf = GaussianDictConfig(
                epochs=cfg.bsesm_params.dict_epochs,
                alpha=1e-3,
                criterion=torch.nn.MSELoss(),
                optimizer_factory=adamw_factory,
                mu_epochs=10,
                rho_epochs=10,
                split_mu_rho=False,
                eig_range=[0.05, 0.2],
                mu_range=[0.0, 1.0],
                regularization_func=(
                    GaussianDictLayer.electrostatic_regularization
                ),
                regularization_gamma=1e-5,
                device=device,
            )

            sc_conf = ISTAConfig(
                epochs=cfg.bsesm_params.sc_epochs,
                alpha=0.1,
                lambd=0.005,
                step_size_method=StepSizeMethod.FROBENIUS,
                power_iterations=10,
                n_functions=n_atoms,
                criterion=torch.nn.MSELoss(),
                device=device,
            )

            if cfg.method.name == "kdtree":

                strategy_conf = KDTreeStrategyConfig(
                    maxNodeSize=cfg.dim * cfg.method.maxNodeSize,
                    device=device,
                    data_wrapper=SESMData,
                )

                part_conf = AdaptivePartitionConfig(
                    overlap_ratio=cfg.method.overlap_ratio,
                    partition_strategy=KDTreeStrategy,
                    strategy_config=strategy_conf,
                )

            else:

                lim_min, lim_max = cfg.dataset.limits

                bounds_tensor = torch.tensor(
                    [
                        [float(lim_min)] * cfg.dim,
                        [float(lim_max)] * cfg.dim,
                    ],
                    dtype=torch.float32,
                    device=device,
                )

                part_conf = UniformPartitionConfig(
                    T=cfg.method.T,
                    initial_bounds=bounds_tensor,
                    activity_threshold=0,
                    overlap_ratio=cfg.method.overlap_ratio,
                    device=device,
                )

            bsesm_conf = BSESMConfig(
                n_features=cfg.dim,
                model_epochs=cfg.bsesm_params.global_epochs,
                partition_config=part_conf,
                dict_config=dict_conf,
                sparse_coding_config=copy.deepcopy(sc_conf),
                log_interval=1000,
                solver_strategy=BSESMSolverStrategy.SEQUENTIAL,
                device=device,
            )

            model = BSESM(config=bsesm_conf, logger=logger)

            previous_n = 0

            for current_n in steps:

                x_batch = x_full_train[previous_n:current_n]
                y_batch = y_full_train[previous_n:current_n]

                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.synchronize(device)

                t0_train = time.time()

                gpu_sampler.start()

                try:
                    model.partial_fit(x_batch, y_batch)
                finally:
                    gpu_sampler.stop()

                if torch.cuda.is_available():
                    torch.cuda.synchronize(device)

                    torch_peak_alloc_mb = (
                        torch.cuda.max_memory_allocated(device)
                        / (1024 ** 2)
                    )

                    torch_peak_reserved_mb = (
                        torch.cuda.max_memory_reserved(device)
                        / (1024 ** 2)
                    )
                else:
                    torch_peak_alloc_mb = 0.0
                    torch_peak_reserved_mb = 0.0

                train_time = time.time() - t0_train

                gpu_stats = gpu_sampler.summary()

                gpu_stats["torch_peak_alloc_mb"] = torch_peak_alloc_mb
                gpu_stats["torch_peak_reserved_mb"] = (
                    torch_peak_reserved_mb
                )

                t0_test = time.time()

                y_pred, _, _ = model.performance_stats(
                    x_test,
                    y_test,
                )

                test_time = time.time() - t0_test

                y_true_cpu = y_test.detach().cpu()
                y_pred_cpu = y_pred.detach().cpu()

                if cfg.dim <= 2 and len(cfg.methods_to_test) > 1:
                    predictions_cache.setdefault(current_n, {})
                    predictions_cache[current_n][method_name] = y_pred_cpu

                mse = mean_squared_error(y_true_cpu, y_pred_cpu)
                mae = mean_absolute_error(y_true_cpu, y_pred_cpu)

                logger.info(
                    (
                        "Step %d | "
                        "MSE: %.5f | "
                        "MAE: %.5f | "
                        "Train: %.2fs | "
                        "Test: %.2fs | "
                        "PeakAlloc: %.2fMB"
                    ),
                    current_n,
                    mse,
                    mae,
                    train_time,
                    test_time,
                    torch_peak_alloc_mb,
                )

                if (
                    cfg.dim <= 2
                    and method_name == cfg.methods_to_test[-1]
                    and len(predictions_cache[current_n]) > 1
                ):

                    try:

                        dataset = {
                            "x_test": x_test.cpu(),
                            "y_test": y_test.cpu(),
                            "x_train": x_full_train[:current_n].cpu(),
                            "y_train": y_full_train[:current_n].cpu(),
                        }

                        plot_name = (
                            f"./outputs/"
                            f"run{run_idx}_"
                            f"step{current_n}_"
                            f"{cfg.dataset.name}_"
                            f"COMPARISON.png"
                        )

                        plot_multi_method_comparison(
                            dataset=dataset,
                            predictions_dict=predictions_cache[current_n],
                            dim=cfg.dim,
                            title=(
                                f"Run {run_idx} "
                                f"N={current_n} "
                                f"{cfg.dataset.name}"
                            ),
                            outpath=plot_name,
                        )

                        if os.path.exists(plot_name):
                            wandb.log(
                                {
                                    "Comparison_Plot": wandb.Image(
                                        plot_name
                                    )
                                }
                            )

                    except Exception as e:
                        logger.info(e)

                previous_n = current_n

            del model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    gpu_sampler.close()

    return "Done"


cfg = {
    "n_runs": 25,
    "seed": 42,
    "dim": 5,
    "stream_steps": 5,

    "dataset": {
        "name": "custom_nd_function",
        "limits": [-1.0, 1.0],
    },

    "methods_to_test": ["kdtree"],

    "bsesm_params": {
        "atoms_per_dim": 10,
        "dict_epochs": 50,
        "sc_epochs": 15,
        "global_epochs": 150,
    },

    "method": {
        "name": "kdtree",
        "maxNodeSize": 10,
        "overlap_ratio": 0.5,
    },
}


logger = logging.getLogger("BSESM")


if __name__ == "__main__":

    cfg = OmegaConf.create(cfg)

    train_stream_experiment(
        cfg,
        logger,
        fun.function_zhou,
    )