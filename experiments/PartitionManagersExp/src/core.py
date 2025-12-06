"""Core training logic used by PartitionManagers experiments.

This module exposes ``train_one_run`` which builds a dataset, configures a
BSESM model and runs a single training + evaluation cycle. It is intentionally
kept as a small runner that integrates components from the local ``pysesm``
package and logs results to Weights & Biases.
"""

import copy
import csv
import os
import time

import hydra
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
from src.utils import plot_multi_method_comparison

# ==========================================
# FUNCIONES DE CHECKPOINT (CSV)
# ==========================================
def get_checkpoint_path(filename="experiment_results.csv"):
    """Obtiene la ruta absoluta al archivo CSV en la raíz del proyecto."""
    try:
        # Intenta obtener la ruta original desde donde se lanzó Hydra
        base_path = hydra.utils.get_original_cwd()
    except:
        base_path = os.getcwd()
    return os.path.join(base_path, filename)

def load_existing_results(filepath):
    """
    Carga los resultados existentes en un Set para búsqueda rápida O(1).
    Retorna: Set con tuplas (run_id, dim, dataset_name, method_name, n_samples)
    """
    completed = set()
    if not os.path.exists(filepath):
        return completed

    try:
        with open(filepath, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convertimos a los tipos correctos para coincidir con el bucle
                key = (
                    int(row['run_id']),
                    int(row['dim']),
                    row['dataset'],
                    row['method'],
                    int(row['n_samples'])
                )
                completed.add(key)
    except Exception as e:
        print(f"[Checkpoint] Advertencia leyendo CSV: {e}")

    return completed

def save_result_row(filepath, data_dict):
    """Guarda una fila de resultados en el CSV (Append mode)."""
    file_exists = os.path.exists(filepath)
    fieldnames = [
        'run_id', 'dim', 'dataset', 'method', 'n_samples', 
        'mse', 'mae', 'inc_time', 'total_time', 'timestamp'
    ]

    try:
        with open(filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

            # Agregamos timestamp
            row = data_dict.copy()
            row['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow(row)
    except Exception as e:
        print(f"[Checkpoint] Error escribiendo en CSV: {e}")

# ==========================================
# LÓGICA PRINCIPAL DEL EXPERIMENTO
# ==========================================
def train_stream_experiment(cfg, logger, func_obj):
    """Run streaming experiment with incremental training and checkpoint support.

    Executes a multi-dimensional streaming experiment that trains BSESM models
    incrementally with different partition strategies (Uniform/KDTree). Supports
    checkpoint-based resumption and logs metrics to Weights & Biases.

    Args:
        cfg: Hydra DictConfig containing experiment parameters including:
            - dim (int): Number of input dimensions
            - n_runs (int): Number of independent runs
            - stream_steps (list): Base step sizes (scaled by dimension)
            - methods_to_test (list): Partition methods ('uniform', 'kdtree')
            - dataset: Dataset configuration (name, limits)
            - bsesm_params: Model hyperparameters
            - seed (int): Random seed base
        logger: Logger instance for progress and error messages.
        func_obj: Callable objective function for synthetic dataset generation.

    Returns:
        str: "Done" upon successful completion of all runs and methods.

    Raises:
        RuntimeError: If shape mismatches occur during model.partial_fit().

    Notes:
        - Steps are scaled exponentially: [base^dim for base in stream_steps]
        - Results are checkpointed to 'experiment_checkpoint.csv'
        - Plots are generated for 2D datasets comparing all methods
        - Models are trained sequentially to maintain state consistency
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- MODIFICACIÓN: Escalar steps según la dimensión ---
    base_steps = cfg.stream_steps
    # Calculamos steps como potencia de la dimensión: [2^d, 4^d, ...]
    steps = [int(n**cfg.dim) for n in base_steps]
    max_samples = max(steps)

    logger.info(f"Stream Steps escalados para Dim {cfg.dim}: {steps}")
    # ------------------------------------------------------

    # 1. PREPARAR CHECKPOINT
    csv_path = get_checkpoint_path("experiment_checkpoint.csv")
    completed_steps = load_existing_results(csv_path)
    logger.info(
        "Checkpoint cargado: %d pasos completados encontrados en %s",
        len(completed_steps), csv_path
    )

    # Obtener ruta base para configs
    try:
        base_path = hydra.utils.get_original_cwd()
    except (ValueError, AttributeError):
        base_path = os.getcwd()

    # ==========================================
    # BUCLE 1: RUNS
    # ==========================================
    for run_idx in range(cfg.n_runs):

        # Caché de predicciones (Volátil, solo para plots en vivo)
        predictions_cache = {}

        current_seed = cfg.seed + run_idx
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)

        logger.info(
            "=== INICIO RUN %d/%d | Dim: %d | Dataset: %s ===",
            run_idx+1, cfg.n_runs, cfg.dim, cfg.dataset.name
        )

        # Generar Dataset (Siempre necesario, usa max_samples actualizado)
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
        # BUCLE 2: MÉTODOS (Secuencial: Uniform -> KDTree)
        # ==========================================
        for method_idx, method_name in enumerate(cfg.methods_to_test):

            logger.info(f"   >>> Configurando Método: {method_name.upper()}")

            # --- Carga de Configuración ---
            try:
                method_conf_path = os.path.join(base_path, "conf", "method", f"{method_name}.yaml")
                if os.path.exists(method_conf_path):
                    method_specific_conf = OmegaConf.load(method_conf_path)
                    cfg.method = method_specific_conf
                else:
                    logger.warning(f"No config for {method_name}, using default.")
            except Exception as e:
                logger.error(f"Error loading config: {e}")
                raise e

            # --- Configuración del Modelo ---
            n_atoms_original = cfg.bsesm_params.atoms_per_dim * cfg.dim

            dict_conf = GaussianDictConfig(
                epochs=cfg.bsesm_params.dict_epochs,
                alpha=1e-3, criterion=torch.nn.MSELoss(),
                optimizer_factory=lambda params, lr: torch.optim.AdamW(params, lr=lr),
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
                    maxNodeSize=cfg.method.maxNodeSize, device=device, data_wrapper=SESMData
                )
                part_conf = AdaptivePartitionConfig(
                    overlap_ratio=cfg.method.overlap_ratio, partition_strategy=KDTreeStrategy,
                    strategy_config=strategy_conf
                )
            else: # Uniform
                lim_min, lim_max = cfg.dataset.limits
                bounds_tensor = torch.tensor([
                    [float(lim_min)] * cfg.dim,
                    [float(lim_max)] * cfg.dim
                ], dtype=torch.float32, device=device)

                part_conf = UniformPartitionConfig(
                    T=cfg.method.T,
                    initial_bounds=bounds_tensor,
                    activity_threshold=0,
                    overlap_ratio=cfg.method.overlap_ratio,
                    device=device
                )

            bsesm_conf = BSESMConfig(
                n_features=cfg.dim, model_epochs=cfg.bsesm_params.global_epochs,
                partition_config=part_conf, dict_config=dict_conf,
                sparse_coding_config=copy.deepcopy(sc_conf),
                log_interval=1000, device=device
            )

            model = BSESM(config=bsesm_conf, logger=logger)

            # ==========================================
            # BUCLE 3: STREAMING (Steps)
            # ==========================================
            previous_n = 0
            total_train_time = 0

            for step_idx, current_n in enumerate(steps):

                # Definir la llave única de este paso
                step_key = (run_idx, cfg.dim, cfg.dataset.name, method_name, current_n)
                is_already_done = step_key in completed_steps

                x_batch = x_full_train[previous_n:current_n]
                y_batch = y_full_train[previous_n:current_n]

                t0 = time.time()

                # --- PARCHE DE SEGURIDAD (Reinicio de n_functions) ---
                if hasattr(model, 'sparse_coding_config'):
                    model.sparse_coding_config.n_functions = n_atoms_original

                # 1. ENTRENAMIENTO (Siempre se ejecuta para mantener el estado del modelo)
                try:
                    model.partial_fit(x_batch, y_batch)
                except RuntimeError as re:
                    logger.critical(f"Error Shapes en Step {current_n} ({method_name}).")
                    raise re

                dt = time.time() - t0
                total_train_time += dt

                # 2. EVALUACIÓN Y LOGGING (Solo si NO está en el checkpoint)
                if is_already_done:
                    logger.info(f"      Step {current_n} | {method_name} | SKIPPING EVAL (Found in Checkpoint)")
                    # Recuperamos el tiempo total teórico del CSV si fuera necesario,
                    # pero como partial_fit se ejecutó, total_train_time es real y correcto.
                else:
                    # Evaluación
                    y_pred, _, _ = model.performance_stats(x_test, y_test)
                    y_true_cpu = y_test.detach().cpu()
                    y_pred_cpu = y_pred.detach().cpu()

                    # Guardar predicción en caché RAM (para plots)
                    if current_n not in predictions_cache:
                        predictions_cache[current_n] = {}
                    predictions_cache[current_n][method_name] = y_pred_cpu

                    mse = mean_squared_error(y_true_cpu, y_pred_cpu)
                    mae = mean_absolute_error(y_true_cpu, y_pred_cpu)

                    logger.info(f"      Step {current_n} | {method_name} | MSE: {mse:.5f}")

                    # --- GUARDAR EN CHECKPOINT CSV ---
                    result_data = {
                        'run_id': run_idx,
                        'dim': cfg.dim,
                        'dataset': cfg.dataset.name,
                        'method': method_name,
                        'n_samples': current_n,
                        'mse': mse,
                        'mae': mae,
                        'inc_time': dt,
                        'total_time': total_train_time
                    }
                    save_result_row(csv_path, result_data)
                    # Agregar a memoria para no repetir si falla en el mismo run
                    completed_steps.add(step_key)

                    # --- LÓGICA DE PLOTEO (Igual que antes) ---
                    plot_image = None
                    is_last_method = (method_idx == len(cfg.methods_to_test) - 1)

                    if cfg.dim == 2 and is_last_method:
                        # Verificamos si tenemos datos para comparar
                        preds_for_step = predictions_cache.get(current_n, {})

                        if preds_for_step:
                            plot_name = f"./outputs/run{run_idx}_step{current_n}_{cfg.dataset.name}_COMPARISON.png"
                            try:
                                x_train_acc = x_full_train[:current_n].cpu()
                                y_train_acc = y_full_train[:current_n].cpu()

                                plot_multi_method_comparison(
                                    X_test=x_test.cpu(),
                                    y_test=y_true_cpu,
                                    predictions_dict=preds_for_step,
                                    X_train=x_train_acc,
                                    y_train=y_train_acc,
                                    dim=cfg.dim,
                                    title=f"Run {run_idx} | N={current_n} | {cfg.dataset.name}",
                                    outpath=plot_name
                                )
                                if os.path.exists(plot_name):
                                    plot_image = wandb.Image(plot_name)
                            except Exception: pass

                    # Log WandB
                    wandb.log({
                        "run_id": run_idx,
                        "n_samples_seen": current_n,
                        "dim": cfg.dim,
                        "dataset": cfg.dataset.name,
                        "method": method_name,
                        "MSE": mse,
                        "MAE": mae,
                        "Incremental_Time": dt,
                        "Total_Train_Time": total_train_time,
                        "Comparison_Plot": plot_image 
                    })

                previous_n = current_n

            # Limpieza
            del model
            torch.cuda.empty_cache()

    return "Done"
