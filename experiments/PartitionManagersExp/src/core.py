
import time
import logging
from src.utils import plot_predictions
import torch

import wandb

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



def train_one_run(cfg, logger, func_obj):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. GENERAR DATASET
    dataset_config = {
        "n_samples": cfg.n_samples,
        "n_dimensions": cfg.dim,
        "function": func_obj,
        "limits": cfg.dataset.limits
    }
    
    train_data, _, _, test_data, _, _ = generate_custom_nd_function_dataset(**dataset_config)
    
    X_train, y_train = train_data["X"], train_data["Z"]
    X_test, y_test = test_data["X"], test_data["Z"]

    # Normalización
    mean_x, std_x = X_train.mean(0), X_train.std(0)
    mean_y, std_y = y_train.mean(), y_train.std()
    
    # Safety checks
    std_x[std_x == 0] = 1.0
    if std_y == 0: std_y = 1.0

    X_train_norm = ((X_train - mean_x) / std_x).to(device)
    y_train_norm = ((y_train - mean_y) / std_y).to(device)
    X_test_norm = ((X_test - mean_x) / std_x).to(device)
    
    # --- CORRECCIÓN IMPORTANTE: Preparamos y_test_norm para pasarlo al modelo ---
    y_test_norm = ((y_test - mean_y) / std_y).to(device)

    # 2. CONFIGURAR MODELO
    # Dictionary
    dict_conf = GaussianDictConfig(
        epochs=cfg.bsesm_params.dict_epochs,
        alpha=1e-3,
        criterion=torch.nn.MSELoss(),
        optimizer_factory=lambda params, lr: torch.optim.AdamW(params, lr=lr),
        mu_epochs=10, rho_epochs=10, split_mu_rho=False,
        eig_range=[0.05, 0.2], mu_range=[0.0, 1.0],
        regularization_func=GaussianDictLayer.electrostatic_regularization,
        regularization_gamma=1e-5,
        device=device
    )

    # Sparse Coding
    n_atoms = cfg.bsesm_params.atoms_per_dim * cfg.dim 
    sc_conf = ISTAConfig(
        epochs=cfg.bsesm_params.sc_epochs,
        alpha=0.1, lambd=0.005,
        step_size_method=StepSizeMethod.FROBENIUS,
        power_iterations=10, n_functions=n_atoms,
        criterion=torch.nn.MSELoss(), device=device
    )

    # Partition Method
    if cfg.method.name == "kdtree":
        strategy_conf = KDTreeStrategyConfig(
            maxNodeSize=cfg.method.maxNodeSize,
            device=device, data_wrapper=SESMData
        )
        part_conf = AdaptivePartitionConfig(
            overlap_ratio=cfg.method.overlap_ratio,
            partition_strategy=KDTreeStrategy,
            strategy_config=strategy_conf
        )
    elif cfg.method.name == "uniform":
        bounds = torch.tensor([[-3.0]*cfg.dim, [3.0]*cfg.dim], device=device)
        part_conf = UniformPartitionConfig(
            T=cfg.method.T,
            initial_bounds=bounds,
            activity_threshold=0,
            overlap_ratio=cfg.method.overlap_ratio,
            device=device
        )
    else:
        raise ValueError(f"Method {cfg.method.name} unknown")

    # BSESM Global
    bsesm_conf = BSESMConfig(
        n_features=cfg.dim,
        model_epochs=cfg.bsesm_params.global_epochs,
        partition_config=part_conf,
        dict_config=dict_conf,
        sparse_coding_config=sc_conf,
        log_interval=50, device=device
    )

    # 3. ENTRENAMIENTO
    model = BSESM(config=bsesm_conf, logger=logger)
    
    t0 = time.time()
    model.partial_fit(X_train_norm, y_train_norm)
    train_time = time.time() - t0

    # 4. TEST
    # --- CORRECCIÓN: Pasamos y_test_norm en lugar de None ---
    y_pred_norm, _, _ = model.performance_stats(X_test_norm, y_test_norm)
    
    # Desnormalizar
    y_pred = y_pred_norm.detach().cpu() * std_y + mean_y
    y_true = y_test.detach().cpu()

    # Métricas
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    # Plot
    plot_name = f"plot_{cfg.dataset.name}_{cfg.method.name}.png"
    plot_predictions(y_true, y_pred, f"{cfg.method.name} - {cfg.dataset.name}", plot_name)

    results = {
        "dim": cfg.dim,
        "n_samples": cfg.n_samples,
        "method": cfg.method.name,
        "dataset": cfg.dataset.name,
        "MSE": mse,
        "MAE": mae,
        "Train_Time": train_time,
        "Plot": wandb.Image(plot_name)
    }
    
    wandb.log(results)
    return results