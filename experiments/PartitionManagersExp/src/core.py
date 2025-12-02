import torch
import time
import wandb
import numpy as np
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
    
    for run_idx in range(cfg.n_runs):
        
        current_seed = cfg.seed + run_idx
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)
        
        logger.info(f"--- Run {run_idx+1}/{cfg.n_runs} | Dim: {cfg.dim} | Dataset: {cfg.dataset.name} ---")

        # 1. GENERAR DATASET (RAW)
        dataset_config = {
            "n_samples": max_samples,
            "n_dimensions": cfg.dim,
            "function": func_obj,
            "limits": cfg.dataset.limits
        }
        
        train_data, _, _, test_data, _, _ = generate_custom_nd_function_dataset(**dataset_config)
        
        # --- SIN NORMALIZAR (Raw Data) ---
        X_full_train = train_data["X"].to(device)
        y_full_train = train_data["Z"].to(device)
        
        X_test = test_data["X"].to(device)
        y_test = test_data["Z"].to(device)

        # 2. CONFIGURACIÓN DEL MODELO
        # Calculamos n_atoms
        n_atoms = cfg.bsesm_params.atoms_per_dim * cfg.dim
        logger.info(f"   Configurando modelo con n_atoms = {n_atoms} (Feature Space)")

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

        # Partición (CRÍTICO: Bounds correctos)
        if cfg.method.name == "kdtree":
            strategy_conf = KDTreeStrategyConfig(
                maxNodeSize=cfg.method.maxNodeSize, device=device, data_wrapper=SESMData
            )
            part_conf = AdaptivePartitionConfig(
                overlap_ratio=cfg.method.overlap_ratio, partition_strategy=KDTreeStrategy,
                strategy_config=strategy_conf
            )
        else: # Uniform
            # --- CORRECCIÓN CLAVE ---
            # Si no normalizamos, los bounds deben ser los límites REALES del dataset.
            # cfg.dataset.limits viene como lista [min, max], ej: [0, 1] o [-5, 5]
            lim_min, lim_max = cfg.dataset.limits
            
            # Construimos tensor de bounds: [[min, min...], [max, max...]]
            bounds_tensor = torch.tensor([
                [float(lim_min)] * cfg.dim, 
                [float(lim_max)] * cfg.dim
            ], dtype=torch.float32, device=device)
            
            part_conf = UniformPartitionConfig(
                T=cfg.method.T, 
                initial_bounds=bounds_tensor, # <--- AHORA USA LÍMITES REALES
                activity_threshold=0,
                overlap_ratio=cfg.method.overlap_ratio, 
                device=device
            )

        bsesm_conf = BSESMConfig(
            n_features=cfg.dim, model_epochs=cfg.bsesm_params.global_epochs,
            partition_config=part_conf, dict_config=dict_conf,
            sparse_coding_config=sc_conf, log_interval=1000, device=device
        )

        # Inicializar
        model = BSESM(config=bsesm_conf, logger=logger)

        # 3. STREAMING LOOP
        previous_n = 0
        total_train_time = 0
        
        for step_idx, current_n in enumerate(steps):
            
            # Slicing de datos raw
            X_batch = X_full_train[previous_n:current_n]
            y_batch = y_full_train[previous_n:current_n]
            
            # Entrenamiento incremental
            t0 = time.time()
            model.partial_fit(X_batch, y_batch) # Pasamos raw data
            dt = time.time() - t0
            total_train_time += dt
            
            # Evaluación
            # Pasamos y_test real como target
            y_pred, _, _ = model.performance_stats(X_test, y_test)
            
            # Métricas (Aseguramos CPU para sklearn)
            y_true_cpu = y_test.detach().cpu()
            y_pred_cpu = y_pred.detach().cpu()
            
            mse = mean_squared_error(y_true_cpu, y_pred_cpu)
            mae = mean_absolute_error(y_true_cpu, y_pred_cpu)
            
            logger.info(f"   Step {current_n} samples | MSE: {mse:.5f}")

            # Plotting (Solo último paso y dimensiones bajas)
            plot_image = None
            
            if cfg.dim in [2]:
                plot_name = f"./outputs/run{run_idx}_step{current_n}_{cfg.dataset.name}_{cfg.method.name}.png"
                
                # Datos acumulados para mostrar puntos de entrenamiento
                X_train_acc = X_full_train[:current_n].cpu()
                y_train_acc = y_full_train[:current_n].cpu()
                
                plot_surface_comparison(
                    X_test=X_test.cpu(), y_test=y_true_cpu, y_pred=y_pred_cpu,
                    X_train=X_train_acc, y_train=y_train_acc,
                    dim=cfg.dim, 
                    title=f"Run {run_idx} | N={current_n} | {cfg.dataset.name}", 
                    outpath=plot_name
                )
                
                import os
                if os.path.exists(plot_name):
                    plot_image = wandb.Image(plot_name)

            # Log WandB
            wandb.log({
                "run_id": run_idx,
                "n_samples_seen": current_n,
                "dim": cfg.dim,
                "dataset": cfg.dataset.name,
                "method": cfg.method.name,
                "MSE": mse,
                "MAE": mae,
                "Incremental_Time": dt,
                "Total_Train_Time": total_train_time,
                "Surface_Plot": plot_image
            })
            
            previous_n = current_n

    return "Done"