"""Lógica de entrenamiento del experimento Ring Oscillator (SESM).

Diseño (validez estadística — trainings independientes):
    - Validación: UN CSV fijo, compartido por todos los runs.
    - Entrenamiento: UN CSV POR (run, size). Cada combinación genera EXACTAMENTE
      `size` muestras SPICE frescas (seed = base_seed + run*100000 + size) y
      entrena un BSESM desde cero. No hay reutilización de datos entre sizes.
    - Dentro de un (run, size), TODOS los métodos comparten esos datos
      → comparación pareada uniform vs kdtree.
    - TargetScaler se ajusta sobre validación → métricas comparables entre runs.
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

from generate_ring_osc_dataset import generate_ring_osc_dataset

from src.utils import (
    GPURAMStepSampler,
    TargetScaler,
    fit_feature_scaler,
    load_existing_results,
    save_result_row,
    scale_features,
)

FEATURE_COLUMNS = ['W_n', 'W_p', 'L', 'Vdd', 'C_load']
RUN_SEED_STRIDE = 100_000  # asegura no-colisión de seeds entre runs y sizes


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
        raise FileNotFoundError(f"CSV no encontrado: {csv_path}")
    df = pd.read_csv(csv_path)
    needed = FEATURE_COLUMNS + [target]
    df = df.dropna(subset=needed).reset_index(drop=True)
    if target in ('f_osc', 'P_avg', 't_rise'):
        df = df[df[target] > 0].reset_index(drop=True)
    return df


def _build_method_config(method_name, method_cfg, dim, device):
    """AdaptivePartitionConfig / UniformPartitionConfig para BSESM.
    Features normalizadas → bounds [0,1] en todas las dimensiones.
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


def _seed_for(cfg, run_idx: int, n_samples: int) -> int:
    """Seed reproducible y único por (run_idx, n_samples)."""
    return int(cfg.training_per_run.base_seed) + run_idx * RUN_SEED_STRIDE + int(n_samples)


def _ensure_run_size_csv(cfg, run_idx: int, n_samples: int,
                         logger: logging.Logger) -> str:
    """Genera (o resume) el CSV de entrenamiento para (run_idx, n_samples).
    Devuelve la ruta absoluta. El archivo contendrá exactamente n_samples
    simulaciones SPICE (algunas pueden ser NaN; se filtran al cargar).
    """
    trn = cfg.training_per_run
    osc = cfg.oscillator

    csv_dir = _resolve_path(str(trn.csv_dir))
    os.makedirs(csv_dir, exist_ok=True)

    pattern = str(trn.get("csv_pattern",
                          "run_{run:02d}_n_{n_samples:05d}.csv"))
    csv_path = os.path.join(csv_dir, pattern.format(run=run_idx,
                                                    n_samples=n_samples))

    seed = _seed_for(cfg, run_idx, n_samples)
    logger.info("RUN %d | size=%d | CSV=%s | seed=%d",
                run_idx, n_samples, csv_path, seed)

    generate_ring_osc_dataset(
        n_samples=n_samples,
        csv_path=csv_path,
        seed=seed,
        plot_every=int(trn.get("plot_every", 0)),
        plot_dir=trn.get("plot_dir", "plots_train"),
        n_stages=int(osc.n_stages),
        nmos_params=OmegaConf.to_container(osc.nmos, resolve=True),
        pmos_params=OmegaConf.to_container(osc.pmos, resolve=True),
        ranges=OmegaConf.to_container(osc.ranges, resolve=True),
        resume=True,
        flush_every=10,
        progress_log=True,
    )
    return csv_path


def _train_eval_once(model, x_train, y_train, x_val, y_val_norm,
                     gpu_sampler, device):
    """Entrena `model` con (x_train, y_train) y evalúa contra validación.
    Devuelve (train_time, test_time, peak_alloc, peak_reserved, y_pred_norm).
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    t0 = time.time()
    gpu_sampler.start()
    try:
        model.partial_fit(x_train, y_train)
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

    t0 = time.time()
    y_pred_norm, _, _ = model.performance_stats(x_val, y_val_norm)
    test_time = time.time() - t0

    return train_time, test_time, peak_alloc, peak_reserved, y_pred_norm


def _build_bsesm(cfg, method_name, method_cfg, dim, n_atoms, device, logger):
    """Construye un BSESM nuevo (fresh model) para un entrenamiento independiente."""
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
    return BSESM(config=bsesm_conf, logger=logger)


def train_ring_osc_experiment(cfg, logger: logging.Logger):
    """Barrido n_runs × training_sizes × methods con BSESM (trainings independientes).

    Cada (run, size) → CSV propio con n_samples frescos. Todos los métodos
    de cfg.methods_to_test se entrenan sobre los MISMOS datos (comparación pareada).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = len(FEATURE_COLUMNS)
    target = cfg.target

    # ---------- Validación ----------
    val_path = _resolve_path(cfg.validation.csv_path)
    logger.info("Validación: %s", val_path)
    df_val = _load_clean(val_path, target)
    logger.info("Validación filas válidas: %d", len(df_val))

    # ---------- Normalización ----------
    ranges = OmegaConf.to_container(cfg.oscillator.ranges, resolve=True)
    mins, maxs = fit_feature_scaler(ranges, FEATURE_COLUMNS)
    mins = mins.to(device)
    maxs = maxs.to(device)

    target_scaler = TargetScaler(use_log=bool(cfg.target_log))
    target_scaler.fit(torch.from_numpy(df_val[target].values.astype(np.float32)))
    logger.info("Target scaler (fit on validation) | log=%s | mean=%.6f | std=%.6f",
                target_scaler.use_log, target_scaler.mean, target_scaler.std)

    def to_tensors(df: pd.DataFrame):
        x_raw = torch.from_numpy(df[FEATURE_COLUMNS].values.astype(np.float32)).to(device)
        y_raw = torch.from_numpy(df[target].values.astype(np.float32)).to(device)
        x_norm = scale_features(x_raw, mins, maxs).clamp(0.0, 1.0)
        y_norm = target_scaler.transform(y_raw)
        return x_norm, y_norm, y_raw

    x_val, y_val_norm, y_val_raw = to_tensors(df_val)

    training_sizes = sorted({int(s) for s in cfg.training_sizes})
    if not training_sizes:
        raise ValueError("cfg.training_sizes está vacío")
    logger.info("Training sizes: %s", training_sizes)

    # ---------- Recovery ----------
    results_csv = _resolve_path(cfg.output.results_csv)
    completed = (load_existing_results(results_csv)
                 if cfg.get("recovery_mode", True) else set())
    if completed:
        logger.info("Recovery: %d filas ya completadas en %s",
                    len(completed), results_csv)

    gpu_sampler = GPURAMStepSampler(
        enabled=True,
        sample_interval_sec=0.5,
        logger=logger,
        device_index=(torch.cuda.current_device()
                      if torch.cuda.is_available() else 0),
    )

    try:
        base_path = hydra.utils.get_original_cwd()
    except (ValueError, AttributeError):
        base_path = os.getcwd()

    n_atoms = cfg.bsesm_params.atoms_per_dim * dim
    y_val_raw_cpu = y_val_raw.detach().cpu()

    # Pre-cargar configs de método (las mismas para todos los runs/sizes)
    method_cfgs = {}
    for method_name in cfg.methods_to_test:
        path = os.path.join(base_path, "conf", "method", f"{method_name}.yaml")
        if not os.path.exists(path):
            raise FileNotFoundError(f"conf/method/{method_name}.yaml no encontrado")
        method_cfgs[method_name] = OmegaConf.load(path)

    # ---------- Bucle principal ----------
    for run_idx in range(cfg.n_runs):
        seed = cfg.seed + run_idx
        torch.manual_seed(seed)
        np.random.seed(seed)

        logger.info("=== RUN %d/%d | target=%s | dim=%d ===",
                    run_idx + 1, cfg.n_runs, target, dim)

        for size in training_sizes:
            # ¿Toda la combinación (run, size) ya está hecha para todos los métodos?
            tuple_done = all(
                (run_idx, target, method_name, size) in completed
                for method_name in cfg.methods_to_test
            )
            if tuple_done:
                logger.info("  (run=%d, size=%d) ya completo para todos los métodos — saltando.",
                            run_idx, size)
                continue

            # 1) Generar / resumir CSV propio (datos compartidos por los métodos)
            csv_path = _ensure_run_size_csv(cfg, run_idx, size, logger)
            df_run = _load_clean(csv_path, target)
            n_valid = len(df_run)
            if n_valid == 0:
                logger.error("  (run=%d, size=%d) sin filas válidas — saltando.",
                             run_idx, size)
                continue
            if n_valid < size:
                logger.warning(
                    "  (run=%d, size=%d) solo %d filas válidas tras NaN — "
                    "se entrena con esas.",
                    run_idx, size, n_valid,
                )

            x_train, y_train_norm, _ = to_tensors(df_run)

            # 2) Entrenar cada método sobre los MISMOS datos
            for method_name in cfg.methods_to_test:
                step_key = (run_idx, target, method_name, size)
                if step_key in completed:
                    logger.info("    método %s ya completo (run=%d, size=%d).",
                                method_name, run_idx, size)
                    continue

                logger.info("  >>> run=%d | size=%d | método=%s",
                            run_idx, size, method_name.upper())

                model = _build_bsesm(cfg, method_name, method_cfgs[method_name],
                                     dim, n_atoms, device, logger)

                train_time, test_time, peak_alloc, peak_reserved, y_pred_norm = \
                    _train_eval_once(model, x_train, y_train_norm,
                                     x_val, y_val_norm, gpu_sampler, device)

                y_pred_norm_cpu = y_pred_norm.detach().cpu()
                y_true_norm_cpu = y_val_norm.detach().cpu()
                mse_norm = float(mean_squared_error(y_true_norm_cpu, y_pred_norm_cpu))
                mae_norm = float(mean_absolute_error(y_true_norm_cpu, y_pred_norm_cpu))

                y_pred_raw = target_scaler.inverse(y_pred_norm_cpu)
                mse_orig = float(mean_squared_error(y_val_raw_cpu, y_pred_raw))
                mae_orig = float(mean_absolute_error(y_val_raw_cpu, y_pred_raw))

                tele = gpu_sampler.summary()
                tele["torch_peak_alloc_mb"] = peak_alloc
                tele["torch_peak_reserved_mb"] = peak_reserved

                logger.info(
                    "    n=%d | MSE_norm=%.5f | MAE_orig=%.3e | train=%.2fs | test=%.2fs",
                    size, mse_norm, mae_orig, train_time, test_time,
                )

                save_result_row(results_csv, {
                    'run_id': run_idx,
                    'target': target,
                    'method': method_name,
                    'n_samples': size,
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
                        "n_samples": size,
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
                    pass

                completed.add(step_key)

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    gpu_sampler.close()
    return "Done"