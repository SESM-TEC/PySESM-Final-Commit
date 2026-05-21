"""Lógica de entrenamiento del experimento Ring Oscillator (SESM).

Esta función `train_ring_osc_experiment` replica el patrón de
PartitionManagersExp pero adaptado para:
    - 5 features físicas (W_n, W_p, L, Vdd, C_load)
    - 1 target (configurable; típicamente f_osc)
    - Datasets generados con PySpice (no analíticos)
    - Streaming cumulativo: cada tamaño se entrena partial_fit del delta.

Se asume que los CSV de pool de entrenamiento y validación ya existen
(generados con generate_validation_dataset.py / generate_training_pool).
"""

import copy
import logging
import os
import time

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.metrics import mean_absolute_error, mean_squared_error

import wandb

from pysesm.blocks.AdaptivePartitionManager import AdaptivePartitionConfig
from pysesm.blocks.KDTreeStrategy import KDTreeStrategy, KDTreeStrategyConfig
from pysesm.blocks.SESMData import SESMData
from pysesm.blocks.UniformPartitionManager import UniformPartitionConfig
from pysesm.dictionaries import GaussianDictConfig, GaussianDictLayer
from pysesm.models.BSESM import BSESM, BSESMConfig, BSESMSolverStrategy
from pysesm.sparse_coding import ISTAConfig, StepSizeMethod

# Cargado relativo dentro del paquete del experimento (DoE/src/)
from src.utils import (
    GPURAMStepSampler,
    TargetScaler,
    fit_feature_scaler,
    load_existing_results,
    save_result_row,
    scale_features,
)

FEATURE_COLUMNS = ['W_n', 'W_p', 'L', 'Vdd', 'C_load']


def _adamw_factory(params, lr):
    return torch.optim.AdamW(params, lr=lr)


def _resolve_path(p: str) -> str:
    if os.path.isabs(p):
        return p
    try:
        base = hydra.utils.get_original_cwd()
    except (ValueError, AttributeError):
        base = os.getcwd()
    return os.path.join(base, p)


def _load_clean(csv_path: str, target: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV no encontrado: {csv_path}. "
            f"Genéralo primero (generate_validation_dataset.py o equivalente)."
        )
    df = pd.read_csv(csv_path)
    needed = FEATURE_COLUMNS + [target]
    df = df.dropna(subset=needed).reset_index(drop=True)
    if target == 'f_osc' or target == 'P_avg' or target == 't_rise':
        # Filtra valores no-positivos (necesario si target_log=True; barato si no).
        df = df[df[target] > 0].reset_index(drop=True)
    return df


def _build_method_config(method_name, method_cfg, dim, device):
    """Devuelve el AdaptivePartitionConfig / UniformPartitionConfig para BSESM.

    Las features están normalizadas a [0,1] en todas las dimensiones.
    """
    if method_name == "kdtree":
        strategy_conf = KDTreeStrategyConfig(
            maxNodeSize=dim * method_cfg.maxNodeSize,
            device=device,
            data_wrapper=SESMData,
        )
        return AdaptivePartitionConfig(
            overlap_ratio=method_cfg.overlap_ratio,
            partition_strategy=KDTreeStrategy,
            strategy_config=strategy_conf,
        )

    bounds = torch.tensor(
        [[0.0] * dim, [1.0] * dim],
        dtype=torch.float32,
        device=device,
    )
    return UniformPartitionConfig(
        T=method_cfg.T,
        initial_bounds=bounds,
        activity_threshold=0,
        overlap_ratio=method_cfg.overlap_ratio,
        device=device,
    )


def train_ring_osc_experiment(cfg, logger: logging.Logger):
    """Ejecuta el barrido n_runs x training_sizes x methods sobre el dataset
    del oscilador, evaluando contra el CSV de validación.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = len(FEATURE_COLUMNS)
    target = cfg.target

    # ---------- Cargar datasets (raw) ----------
    val_path = _resolve_path(cfg.validation.csv_path)
    pool_path = _resolve_path(cfg.training_pool.csv_path)
    logger.info("Validación: %s", val_path)
    logger.info("Pool entrenamiento: %s", pool_path)

    df_val = _load_clean(val_path, target)
    df_pool = _load_clean(pool_path, target)
    logger.info("Filas válidas | pool=%d | val=%d", len(df_pool), len(df_val))

    # ---------- Normalización ----------
    ranges = OmegaConf.to_container(cfg.oscillator.ranges, resolve=True)
    mins, maxs = fit_feature_scaler(ranges, FEATURE_COLUMNS)
    mins = mins.to(device)
    maxs = maxs.to(device)

    target_scaler = TargetScaler(use_log=bool(cfg.target_log))
    target_scaler.fit(torch.from_numpy(df_pool[target].values.astype(np.float32)))

    logger.info("Target scaler | log=%s | mean=%.6f | std=%.6f",
                target_scaler.use_log, target_scaler.mean, target_scaler.std)

    def to_tensors(df: pd.DataFrame):
        x_raw = torch.from_numpy(df[FEATURE_COLUMNS].values.astype(np.float32)).to(device)
        y_raw = torch.from_numpy(df[target].values.astype(np.float32)).to(device)
        x_norm = scale_features(x_raw, mins, maxs).clamp(0.0, 1.0)
        y_norm = target_scaler.transform(y_raw)
        return x_norm, y_norm, y_raw

    x_val, y_val_norm, y_val_raw = to_tensors(df_val)
    x_pool, y_pool_norm, _ = to_tensors(df_pool)

    # Sizes válidos = solo los <= filas disponibles tras dropna
    training_sizes = sorted({int(s) for s in cfg.training_sizes if int(s) <= len(df_pool)})
    if not training_sizes:
        raise ValueError(
            f"Ningún training_size <= {len(df_pool)} filas válidas en el pool."
        )
    max_train = max(training_sizes)
    logger.info("Training sizes: %s", training_sizes)

    # ---------- Recovery ----------
    results_csv = _resolve_path(cfg.output.results_csv)
    completed = (load_existing_results(results_csv)
                 if cfg.get("recovery_mode", True) else set())
    if completed:
        logger.info("Recovery: %d filas ya completadas en %s",
                    len(completed), results_csv)

    # Telemetría
    gpu_sampler = GPURAMStepSampler(
        enabled=True,
        sample_interval_sec=0.5,
        logger=logger,
        device_index=(torch.cuda.current_device()
                      if torch.cuda.is_available() else 0),
    )

    # Path base para configs de método
    try:
        base_path = hydra.utils.get_original_cwd()
    except (ValueError, AttributeError):
        base_path = os.getcwd()

    n_atoms = cfg.bsesm_params.atoms_per_dim * dim

    # ---------- Bucle de runs ----------
    for run_idx in range(cfg.n_runs):
        seed = cfg.seed + run_idx
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Permutación reproducible del pool para cada run
        perm = torch.randperm(len(df_pool), generator=torch.Generator().manual_seed(seed))
        x_train_full = x_pool[perm[:max_train]]
        y_train_full = y_pool_norm[perm[:max_train]]

        logger.info("=== RUN %d/%d | target=%s | dim=%d ===",
                    run_idx + 1, cfg.n_runs, target, dim)

        # ---------- Bucle de métodos ----------
        for method_name in cfg.methods_to_test:
            # Saltar si todos los tamaños ya están completos para este método
            method_done = all(
                (run_idx, target, method_name, s) in completed
                for s in training_sizes
            )
            if method_done:
                logger.info("  método %s ya completo — saltando.", method_name)
                continue

            method_cfg_path = os.path.join(
                base_path, "conf", "method", f"{method_name}.yaml"
            )
            if os.path.exists(method_cfg_path):
                method_cfg = OmegaConf.load(method_cfg_path)
            else:
                raise FileNotFoundError(f"conf/method/{method_name}.yaml no encontrado")

            logger.info("  >>> Método: %s", method_name.upper())

            # Configuración BSESM (recreada por método)
            dict_conf = GaussianDictConfig(
                epochs=cfg.bsesm_params.dict_epochs,
                alpha=1e-3,
                criterion=torch.nn.MSELoss(),
                optimizer_factory=_adamw_factory,
                mu_epochs=10, rho_epochs=10, split_mu_rho=False,
                eig_range=[0.05, 0.2], mu_range=[0.0, 1.0],
                regularization_func=GaussianDictLayer.electrostatic_regularization,
                regularization_gamma=1e-5,
                device=device,
            )
            sc_conf = ISTAConfig(
                epochs=cfg.bsesm_params.sc_epochs,
                alpha=0.1, lambd=0.005,
                step_size_method=StepSizeMethod.FROBENIUS,
                power_iterations=10,
                n_functions=n_atoms,
                criterion=torch.nn.MSELoss(),
                device=device,
            )
            part_conf = _build_method_config(method_name, method_cfg, dim, device)
            bsesm_conf = BSESMConfig(
                n_features=dim,
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

            # ---------- Bucle de tamaños (streaming cumulativo) ----------
            for current_n in training_sizes:
                step_key = (run_idx, target, method_name, current_n)
                already_done = step_key in completed

                x_batch = x_train_full[previous_n:current_n]
                y_batch = y_train_full[previous_n:current_n]

                if already_done:
                    logger.info("    n=%d ya completo en CSV — solo actualizo modelo",
                                current_n)
                    model.partial_fit(x_batch, y_batch)
                    previous_n = current_n
                    continue

                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.synchronize(device)

                t0 = time.time()
                gpu_sampler.start()
                try:
                    model.partial_fit(x_batch, y_batch)
                finally:
                    gpu_sampler.stop()
                if torch.cuda.is_available():
                    torch.cuda.synchronize(device)
                    peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
                else:
                    peak_alloc = 0.0
                    peak_reserved = 0.0
                train_time = time.time() - t0

                # Evaluación
                t0 = time.time()
                y_pred_norm, _, _ = model.performance_stats(x_val, y_val_norm)
                test_time = time.time() - t0

                y_true_norm_cpu = y_val_norm.detach().cpu()
                y_pred_norm_cpu = y_pred_norm.detach().cpu()

                mse_norm = float(mean_squared_error(y_true_norm_cpu, y_pred_norm_cpu))
                mae_norm = float(mean_absolute_error(y_true_norm_cpu, y_pred_norm_cpu))

                # Métricas en el espacio original (Hz / W / s)
                y_pred_raw = target_scaler.inverse(y_pred_norm_cpu)
                y_true_raw = y_val_raw.detach().cpu()
                mse_orig = float(mean_squared_error(y_true_raw, y_pred_raw))
                mae_orig = float(mean_absolute_error(y_true_raw, y_pred_raw))

                tele = gpu_sampler.summary()
                tele["torch_peak_alloc_mb"] = peak_alloc
                tele["torch_peak_reserved_mb"] = peak_reserved

                logger.info(
                    "    n=%d | MSE_norm=%.5f | MAE_orig=%.3e | train=%.2fs | test=%.2fs",
                    current_n, mse_norm, mae_orig, train_time, test_time,
                )

                save_result_row(results_csv, {
                    'run_id': run_idx,
                    'target': target,
                    'method': method_name,
                    'n_samples': current_n,
                    'mse_norm': mse_norm,
                    'mae_norm': mae_norm,
                    'mse_orig': mse_orig,
                    'mae_orig': mae_orig,
                    'train_time': train_time,
                    'test_time': test_time,
                    **tele,
                })

                try:
                    wandb.log({
                        "run_id": run_idx,
                        "n_samples": current_n,
                        "target": target,
                        "method": method_name,
                        "MSE_norm": mse_norm,
                        "MAE_norm": mae_norm,
                        "MSE_orig": mse_orig,
                        "MAE_orig": mae_orig,
                        "Train_Time": train_time,
                        "Test_Time": test_time,
                        "Torch_Peak_Alloc_MB": tele["torch_peak_alloc_mb"],
                        "Torch_Peak_Reserved_MB": tele["torch_peak_reserved_mb"],
                        "GPU_Mem_Used_MB_Mean": tele["gpu_mem_used_mb_mean"],
                        "RAM_Used_MB_Mean": tele["ram_used_mb_mean"],
                    })
                except Exception:
                    # W&B puede estar desactivado; no interrumpir el experimento.
                    pass

                completed.add(step_key)
                previous_n = current_n

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    gpu_sampler.close()
    return "Done"
