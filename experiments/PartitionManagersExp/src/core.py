import torch
import time
import wandb
import numpy as np
import os
import hydra
from omegaconf import OmegaConf
from sklearn.metrics import mean_squared_error, mean_absolute_error

# BSESM Imports
from pysesm.models.BSESM import BSESM, BSESMConfig
from pysesm.sparse_coding import ISTAConfig, StepSizeMethod
from pysesm.dictionaries import GaussianDictConfig, GaussianDictLayer
from pysesm.blocks.UniformPartitionManager import UniformPartitionConfig
from pysesm.blocks.AdaptivePartitionManager import AdaptivePartitionConfig
from pysesm.blocks.KDTreeStrategy import KDTreeStrategy, KDTreeStrategyConfig
from pysesm.blocks.SESMData import SESMData
from pysesm.utils_dataset.generate_dataset import generate_custom_nd_function_dataset

from src.utils import plot_surface_comparison

def train_stream_experiment(cfg, logger, func_obj):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Pasos de streaming
    steps = cfg.stream_steps
    max_samples = max(steps)
    
    # Obtener ruta base para cargar configs manualmente
    try:
        base_path = hydra.utils.get_original_cwd()
    except:
        base_path = os.getcwd()

    # ==========================================
    # BUCLE 1: REPETICIONES (RUNS)
    # ==========================================
    for run_idx in range(cfg.n_runs):
        
        # 1. GENERAR DATASET (Se mantiene fijo para TODOS los métodos de este Run)
        current_seed = cfg.seed + run_idx
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)
        
        logger.info(f"=== INICIO RUN {run_idx+1}/{cfg.n_runs} | Dim: {cfg.dim} | Dataset: {cfg.dataset.name} ===")

        dataset_config = {
            "n_samples": max_samples,
            "n_dimensions": cfg.dim,
            "function": func_obj,
            "limits": cfg.dataset.limits
        }
        
        # Generación de datos Raw
        train_data, _, _, test_data, _, _ = generate_custom_nd_function_dataset(**dataset_config)
        
        # Mover a dispositivo
        X_full_train = train_data["X"].to(device)
        y_full_train = train_data["Z"].to(device)
        
        X_test = test_data["X"].to(device)
        y_test = test_data["Z"].to(device)
        
        # ==========================================
        # BUCLE 2: MÉTODOS (Uniform -> KDTree)
        # ==========================================
        for method_name in cfg.methods_to_test:
            
            logger.info(f"   >>> Configurando Método: {method_name.upper()}")

            # --- CARGA DINÁMICA DE CONFIGURACIÓN DEL MÉTODO ---
            try:
                method_conf_path = os.path.join(base_path, "conf", "method", f"{method_name}.yaml")
                if os.path.exists(method_conf_path):
                    method_specific_conf = OmegaConf.load(method_conf_path)
                    cfg.method = method_specific_conf
                else:
                    logger.warning(f"No se encontró {method_conf_path}, usando config por defecto si existe.")
            except Exception as e:
                logger.error(f"Error cargando config de {method_name}: {e}")
                raise e

            # 2. CONFIGURACIÓN DEL MODELO
            n_atoms = cfg.bsesm_params.atoms_per_dim * cfg.dim
            
            # Diccionario
            dict_conf = GaussianDictConfig(
                epochs=cfg.bsesm_params.dict_epochs,
                alpha=1e-3, criterion=torch.nn.MSELoss(),
                optimizer_factory=lambda params, lr: torch.optim.AdamW(params, lr=lr),
                mu_epochs=10, rho_epochs=10, split_mu_rho=False,
                eig_range=[0.05, 0.2], mu_range=[0.0, 1.0], 
                regularization_func=GaussianDictLayer.electrostatic_regularization,
                regularization_gamma=1e-5, device=device
            )

            # Sparse Coding
            sc_conf = ISTAConfig(
                epochs=cfg.bsesm_params.sc_epochs,
                alpha=0.1, lambd=0.005, step_size_method=StepSizeMethod.FROBENIUS,
                power_iterations=10, n_functions=n_atoms,
                criterion=torch.nn.MSELoss(), device=device
            )

            # Partición
            if cfg.method.name == "kdtree":
                strategy_conf = KDTreeStrategyConfig(
                    maxNodeSize=cfg.method.maxNodeSize, device=device, data_wrapper=SESMData
                )
                part_conf = AdaptivePartitionConfig(
                    overlap_ratio=cfg.method.overlap_ratio, partition_strategy=KDTreeStrategy,
                    strategy_config=strategy_conf
                )
            else: # Uniform
                x1lim, x2lim = cfg.dataset.limits
                # Asegurar floats para evitar problemas de tipo
                bounds_tensor = torch.tensor([[x1lim for i in range(cfg.dim)], [x2lim for i in range(cfg.dim)]], dtype=torch.float32)
                
                part_conf = UniformPartitionConfig(
                    T=cfg.method.T, # Debe ser 1 en el YAML para evitar error de shapes
                    initial_bounds=bounds_tensor,
                    activity_threshold=0,
                    overlap_ratio=cfg.method.overlap_ratio, 
                    device=device
                )

            bsesm_conf = BSESMConfig(
                n_features=cfg.dim, model_epochs=cfg.bsesm_params.global_epochs,
                partition_config=part_conf, dict_config=dict_conf,
                sparse_coding_config=sc_conf, log_interval=1000, device=device
            )

            # Inicializar Modelo
            model = BSESM(config=bsesm_conf, logger=logger)

            # ==========================================
            # BUCLE 3: DATA STREAM (Steps)
            # ==========================================
            previous_n = 0
            total_train_time = 0
            
            for step_idx, current_n in enumerate(steps):
                
                # Slicing
                X_batch = X_full_train[previous_n:current_n]
                y_batch = y_full_train[previous_n:current_n]
                
                # Entrenamiento incremental
                t0 = time.time()
                try:
                    model.partial_fit(X_batch, y_batch)
                except RuntimeError as re:
                    logger.error(f"Error en partial_fit Step {current_n} ({method_name}): {re}")
                    raise re
                
                dt = time.time() - t0
                total_train_time += dt
                
                # Evaluación
                y_pred, _, _ = model.performance_stats(X_test, y_test)
                
                y_true_cpu = y_test.detach().cpu()
                y_pred_cpu = y_pred.detach().cpu()
                
                mse = mean_squared_error(y_true_cpu, y_pred_cpu)
                mae = mean_absolute_error(y_true_cpu, y_pred_cpu)
                
                logger.info(f"      Step {current_n} | {method_name} | MSE: {mse:.5f}")

                # Plotting (Optimizado)
                plot_image = None
                if cfg.dim in [2]: 
                    plot_name = f"./outputs/run{run_idx}_step{current_n}_{cfg.dataset.name}_{method_name}.png"
                    
                    X_train_acc = X_full_train[:current_n].cpu()
                    y_train_acc = y_full_train[:current_n].cpu()
                    
                    try:
                        plot_surface_comparison(
                            X_test=X_test.cpu(), y_test=y_true_cpu, y_pred=y_pred_cpu,
                            X_train=X_train_acc, y_train=y_train_acc,
                            dim=cfg.dim, 
                            title=f"Run {run_idx} | {method_name} | N={current_n}", 
                            outpath=plot_name
                        )
                        if os.path.exists(plot_name):
                            plot_image = wandb.Image(plot_name)
                    except Exception as e_plot:
                        # No detener el experimento por un error de ploteo
                        logger.warning(f"Warning: Plot failed - {e_plot}")

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
                    "Surface_Plot": plot_image
                })
                
                previous_n = current_n
            
            # Limpieza explícita
            del model
            torch.cuda.empty_cache()

    return "Done"