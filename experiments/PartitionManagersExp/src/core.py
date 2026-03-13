"""Core training logic used by PartitionManagers experiments.
"""

import copy
import csv
import os
import time

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import mean_squared_error, mean_absolute_error

import wandb

from pysesm.blocks.AdaptivePartitionManager import AdaptivePartitionConfig
from pysesm.blocks.KDTreeStrategy import KDTreeStrategy, KDTreeStrategyConfig
from pysesm.blocks.SESMData import SESMData
from pysesm.blocks.UniformPartitionManager import UniformPartitionConfig
from pysesm.dictionaries import GaussianDictConfig, GaussianDictLayer
from pysesm.models.BSESM import BSESM, BSESMConfig
from pysesm.sparse_coding import ISTAConfig, StepSizeMethod
from pysesm.utils_dataset.generate_dataset import (
    generate_custom_nd_function_dataset,
)

from src.utils import GPUStepSampler, plot_multi_method_comparison


# ==========================================
# HELPERS GLOBALES
# ==========================================
def adamw_factory(params, lr):
    """Fábrica de optimizador."""
    return torch.optim.AdamW(params, lr=lr)


# ==========================================
# FUNCIONES DE MANEJO DE ARCHIVOS (CSV)
# ==========================================
def get_checkpoint_path(filename="experiment_results.csv"):
    """Retorna la ruta absoluta al archivo CSV de resultados.
    
    Args:
        filename (str): Nombre del archivo CSV. Default: 'experiment_results.csv'
    
    Returns:
        str: Ruta absoluta al archivo en el directorio base del proyecto.
    """
    try:
        base_path = hydra.utils.get_original_cwd()
    except (ValueError, AttributeError):
        base_path = os.getcwd()
    return os.path.join(base_path, filename)

def load_existing_results(filepath):
    """Retorna un set con las llaves de pasos completados encontrados en el CSV."""
    completed = set()
    if not os.path.exists(filepath):
        return completed
    try:
        with open(filepath, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Llave única: (run, dim, dataset, method, n_samples)
                key = (
                    int(row['run_id']),
                    int(row['dim']),
                    row['dataset'],
                    row['method'],
                    int(row['n_samples'])
                )
                completed.add(key)
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"[Checkpoint] Advertencia leyendo CSV: {e}")
    return completed

def save_result_row(filepath, data_dict):
    """Guarda una fila de resultados en el CSV."""
    # Verificamos existencia para saber si escribir header
    file_exists = os.path.exists(filepath)

    fieldnames = [
        'run_id', 'dim', 'dataset', 'method', 'n_samples',
        'mse', 'mae', 'train_time', 'test_time',
        'gpu_samples',
        'gpu_mem_used_mb_mean',
        'gpu_mem_used_mb_var',
        'gpu_mem_util_mean',
        'gpu_mem_util_var',
        'timestamp'
    ]
    try:
        with open(filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            row = data_dict.copy()
            row['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow(row)
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"[Checkpoint] Error escribiendo en CSV: {e}")

# ==========================================
# LÓGICA PRINCIPAL DEL EXPERIMENTO
# ==========================================
def train_stream_experiment(cfg, logger, func_obj):  # pylint: disable=too-many-nested-blocks
    """Ejecuta experimento de streaming comparando múltiples métodos de particionamiento.
    
    Args:
        cfg (DictConfig): Configuración Hydra con parámetros del experimento.
        logger (logging.Logger): Logger para mensajes informativos y errores.
        func_obj (Callable): Función objetivo para generación del dataset.
    
    Returns:
        str: 'Done' cuando todos los runs/métodos/pasos se completan.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    default_device_index = torch.cuda.current_device() if torch.cuda.is_available() else 0
    gpu_sampler = GPUStepSampler(
        enabled=True,
        sample_interval_sec=0.5,
        logger=logger,
        device_index=default_device_index,
    )

    base_steps = cfg.stream_steps
    MAX_DATASET_SIZE = 55_000

    steps_raw = [int(n**cfg.dim) for n in base_steps]
    steps = [min(s, MAX_DATASET_SIZE) for s in steps_raw]

    # eliminar duplicados y ordenar
    steps = sorted(set(steps))

    max_samples = max(steps)

    logger.info(f"Stream Steps para Dim {cfg.dim}: {steps}")

    # ========================================================
    # LÓGICA DE RECUPERACIÓN (RECOVERY MODE)
    # ========================================================
    csv_path = get_checkpoint_path("experiment_results.csv")

    # Default True si no está en config
    recovery_enabled = cfg.get("recovery_mode", True)

    if recovery_enabled:
        logger.info("Recovery Mode: ACTIVADO. Buscando progreso en %s", csv_path)
        completed_keys = load_existing_results(csv_path)
        logger.info(" -> Se encontraron %d registros completados.", len(completed_keys))
    else:
        logger.info(
            "Recovery Mode: DESACTIVADO. Eliminando historial previo (Sobreescritura total)."
        )
        completed_keys = set()

        # --- MODIFICACIÓN: Eliminar archivo físico para reiniciar desde 0 ---
        if os.path.exists(csv_path):
            try:
                os.remove(csv_path)
                logger.info(" -> Archivo %s eliminado exitosamente.", csv_path)
            except Exception as e: # pylint: disable=broad-exception-caught
                logger.warning(" -> No se pudo eliminar el archivo %s: %s", csv_path, e)
                # Si no se puede borrar (ej: bloqueado), los datos se mezclarán,
                # pero es lo mejor que podemos hacer sin detener el proceso.

    # Obtener ruta base para configs locales
    try:
        base_path = hydra.utils.get_original_cwd()
    except (ValueError, AttributeError):
        base_path = os.getcwd()

    # ==========================================
    # BUCLE 1: RUNS
    # ==========================================
    # BUCLE 1: RUNS
    # ==========================================
    for run_idx in range(cfg.n_runs):
        predictions_cache = {}

        current_seed = cfg.seed + run_idx
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)
        logger.info(
            "=== INICIO RUN %d/%d | Dim: %d | Dataset: %s ===",
            run_idx+1, cfg.n_runs, cfg.dim, cfg.dataset.name
        )

        # Generar Dataset
        dataset_config = {
            "n_samples": max_samples,
            "n_dimensions": cfg.dim,
            "function": func_obj,
            "limits": cfg.dataset.limits
        }
        train_data, _, _, test_data, _, _ = generate_custom_nd_function_dataset(**dataset_config)
        x_full_train = train_data["X"].to(device)
        y_full_train = train_data["Z"].to(device)
        x_test = test_data["X"].to(device)
        y_test = test_data["Z"].to(device)

        # ==========================================
        # BUCLE 2: MÉTODOS
        # ==========================================
        for method_name in cfg.methods_to_test:

            # Verificar si todo el método ya está hecho (Solo si recovery=True)
            method_all_done = False
            if recovery_enabled and steps:
                method_all_done = all(
                    (run_idx, cfg.dim, cfg.dataset.name, method_name, s) in completed_keys
                    for s in steps
                )

            if method_all_done:
                logger.info(
                    "   >>> Método %s COMPLETADO previamente. Saltando...",
                    method_name.upper()
                )
                continue

            logger.info("   >>> Iniciando Método: %s", method_name.upper())

            # Cargar Config Método
            try:
                method_conf_path = os.path.join(base_path, "conf", "method", f"{method_name}.yaml")
                if os.path.exists(method_conf_path):
                    cfg.method = OmegaConf.load(method_conf_path)
            except Exception as e:
                logger.error(f"Error cargando config de método: {e}")
                raise e

            # Config BSESM
            n_atoms_original = cfg.bsesm_params.atoms_per_dim * cfg.dim

            dict_conf = GaussianDictConfig(
                epochs=cfg.bsesm_params.dict_epochs,
                alpha=1e-3, criterion=torch.nn.MSELoss(),
                optimizer_factory=adamw_factory,
                mu_epochs=10, rho_epochs=10, split_mu_rho=False,
                eig_range=[0.05, 0.2], mu_range=[0.0, 1.0],
                regularization_func=GaussianDictLayer.electrostatic_regularization,
                regularization_gamma=1e-5, device=device
            )

            sc_conf = ISTAConfig(
                epochs=cfg.bsesm_params.sc_epochs,
                alpha=0.1, lambd=0.005, step_size_method=StepSizeMethod.FROBENIUS,
                power_iterations=10, n_functions=n_atoms_original,
                criterion=torch.nn.MSELoss(), device=device
            )

            if cfg.method.name == "kdtree":
                strategy_conf = KDTreeStrategyConfig(
                    maxNodeSize=cfg.method.maxNodeSize,
                    device=device,
                    data_wrapper=SESMData
                )
                part_conf = AdaptivePartitionConfig(
                    overlap_ratio=cfg.method.overlap_ratio,
                    partition_strategy=KDTreeStrategy,
                    strategy_config=strategy_conf
                )
            else:  # Uniform
                lim_min, lim_max = cfg.dataset.limits
                bounds_tensor = torch.tensor([
                    [float(lim_min)] * cfg.dim,
                    [float(lim_max)] * cfg.dim
                ],
                dtype=torch.float32,
                device=device)

                part_conf = UniformPartitionConfig(
                    T=cfg.method.T,
                    initial_bounds=bounds_tensor,
                    activity_threshold=0,
                    overlap_ratio=cfg.method.overlap_ratio,
                    device=device
                )

            bsesm_conf = BSESMConfig(
                n_features=cfg.dim, model_epochs=cfg.bsesm_params.global_epochs,
                partition_config=part_conf,
                dict_config=dict_conf,
                sparse_coding_config=copy.deepcopy(sc_conf),
                log_interval=1000,
                device=device
            )

            model = BSESM(config=bsesm_conf, logger=logger)
            previous_n = 0

            # ==========================================
            # BUCLE 3: STREAMING STEPS
            # ==========================================
            for current_n in steps:
                step_key = (run_idx, cfg.dim, cfg.dataset.name, method_name, current_n)
                is_done = step_key in completed_keys

                x_batch = x_full_train[previous_n:current_n]
                y_batch = y_full_train[previous_n:current_n]

                if not is_done:
                    # CASO A: Entrenar y Evaluar
                    t0_train = time.time()
                    gpu_sampler.start()
                    try:
                        model.partial_fit(x_batch, y_batch)
                    finally:
                        gpu_sampler.stop()
                    t_train = time.time() - t0_train
                    gpu_stats = gpu_sampler.summary()

                    t0_test = time.time()
                    y_pred, _, _ = model.performance_stats(x_test, y_test)
                    t_test = time.time() - t0_test

                    # Métricas
                    y_true_cpu = y_test.detach().cpu()
                    y_pred_cpu = y_pred.detach().cpu()

                    if current_n not in predictions_cache:
                        predictions_cache[current_n] = {}
                    predictions_cache[current_n][method_name] = y_pred_cpu

                    mse = mean_squared_error(y_true_cpu, y_pred_cpu)
                    mae = mean_absolute_error(y_true_cpu, y_pred_cpu)

                    logger.info(
                        "      Step %d | MSE: %.5f | Train: %.2fs",
                        current_n, mse, t_train
                    )

                    # CSV (save_result_row creará el archivo con headers si lo borramos antes)
                    save_result_row(csv_path, {
                        'run_id': run_idx,
                        'dim': cfg.dim,
                        'dataset': cfg.dataset.name,
                        'method': method_name,
                        'n_samples': current_n,
                        'mse': mse,
                        'mae': mae,
                        'train_time': t_train,
                        'test_time': t_test,
                        **gpu_stats
                    })

                    wandb.log({
                        "run_id": run_idx,
                        "n_samples": current_n,
                        "dim": cfg.dim,
                        "dataset": cfg.dataset.name,
                        "method": method_name,
                        "MSE": mse,
                        "MAE": mae,
                        "Train_Time_Step": t_train,
                        "Test_Time_Step": t_test,
                        "GPU_Samples": gpu_stats["gpu_samples"],
                        "GPU_Mem_Used_MB_Mean": gpu_stats["gpu_mem_used_mb_mean"],
                        "GPU_Mem_Used_MB_Var": gpu_stats["gpu_mem_used_mb_var"],
                        "GPU_Mem_Util_Mean": gpu_stats["gpu_mem_util_mean"],
                        "GPU_Mem_Util_Var": gpu_stats["gpu_mem_util_var"]
                    })

                    completed_keys.add(step_key)

                else:
                    # CASO B: Recuperar estado (No se escribe en CSV)
                    logger.info(
                        "      Step %d | %s | Recuperando estado (Encontrado en CSV)",
                        current_n, method_name
                    )
                    model.partial_fit(x_batch, y_batch)

                previous_n = current_n

            last_step = steps[-1] if steps else None
            if cfg.dim == 2 and last_step in predictions_cache:
                is_last_method = method_name == cfg.methods_to_test[-1]
                if is_last_method and len(predictions_cache[last_step]) > 0:
                    try:
                        dataset  = {
                            "x_test": x_test.cpu(),
                            "y_test": y_test.cpu(),
                            "x_train": x_full_train[:last_step].cpu(),
                            "y_train": y_full_train[:last_step].cpu(),
                        }

                        plot_name = (
                            f"./outputs/run{run_idx}_step{last_step}_"
                            f"{cfg.dataset.name}_COMPARISON.png"
                        )

                        plot_multi_method_comparison(
                            dataset=dataset,
                            predictions_dict=predictions_cache[last_step],
                            dim=cfg.dim,
                            title=f"Run {run_idx} N={last_step} {cfg.dataset.name}",
                            outpath=plot_name
                        )

                        if os.path.exists(plot_name):
                            wandb.log({"Comparison_Plot": wandb.Image(plot_name)})
                    except Exception: # pylint: disable=broad-exception-caught
                        pass

        del model
        torch.cuda.empty_cache()

        gpu_sampler.close()

    return "Done"