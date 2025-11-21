import os
import logging
import torch
import numpy as np
import joblib
import wandb
from experiment import EXPERIMENT
from collections import defaultdict
import fun
from pysesm.utils_dataset.generate_dataset import generate_custom_function_dataset, generate_custom_nd_function_dataset
from LossWrappers import KLDivLossWrapper, JensenShannonLossWrapper
from pysesm.models.SSESM import SSESMConfig
from pysesm.sparse_coding import ISTAConfig, StepSizeMethod
from pysesm.dictionaries import GaussianDictConfig, GaussianDictLayer
from pysesm.blocks.UniformPartitionManager import UniformPartitionConfig
from pysesm.blocks.KDTreeStrategy import KDTreeStrategy, KDTreeStrategyConfig
from pysesm.blocks.SESMData import SESMData
from pysesm.blocks.AdaptativePartitionManager import AdaptativePartitionConfig
from pysesm.utils.metric_loggers import *
from pysesm.utils.loggers import setup_logger
import gc

def load_checkpoint(checkpoint_dir="./checkpoints"):
    """
    Carga el checkpoint si existe.
    Retorna: (all_metrics_dim, progress_tracker, experiment_config_data) o (None, None, None)
    """
    logger=setup_logger(level=logging.DEBUG)
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.joblib")
    
    if os.path.exists(checkpoint_path):
        logger.debug("Checkpoint encontrado en %s. Cargando...", checkpoint_path)
        checkpoint = joblib.load(checkpoint_path)
        logger.debug("Checkpoint cargado. Progreso: %d combinaciones completadas.",len(checkpoint['progress_tracker']))
        return checkpoint['all_metrics_dim'], checkpoint['progress_tracker'], checkpoint.get('experiment_config_data', None)
    else:
        logger.debug("No se encontró checkpoint. Iniciando desde cero.")
        return {}, set(), None


def save_checkpoint(all_metrics_dim, progress_tracker, experiment_config_data, checkpoint_dir="./checkpoints"):
    """
    Guarda el checkpoint con todos los datos necesarios para recuperación.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.joblib")
    logger=setup_logger(level=logging.DEBUG)
    checkpoint = {
        'all_metrics_dim': all_metrics_dim,
        'progress_tracker': progress_tracker,
        'experiment_config_data': experiment_config_data
    }
    
    # Guardar temporalmente y luego renombrar (atomic write)
    temp_path = checkpoint_path + ".tmp"
    joblib.dump(checkpoint, temp_path)
    os.replace(temp_path, checkpoint_path)
    
    logger.debug("Checkpoint guardado: %s combinaciones completadas.",len(progress_tracker))


def is_combination_done(progress_tracker, function_name, dim, n, run_idx):
    """
    Verifica si una combinación específica ya fue completada.
    """
    key = (function_name, dim, n, run_idx)
    return key in progress_tracker


def mark_combination_done(progress_tracker, function_name, dim, n, run_idx):
    """
    Marca una combinación como completada.
    """
    key = (function_name, dim, n, run_idx)
    progress_tracker.add(key)


def main():
    """
    Script para correr múltiples experimentos con sistema de checkpoint/recovery.
    """
    
    logger=setup_logger(level=logging.DEBUG)
    # CONFIGURACIONES DEL EXPERIMENTO
    function_limits = {
        "function_zakharov": [-10, 10],
        "function_styblinski_tang": [-5, 5],
        "function_zhou": [0, 1]
    }
    functions = [fun.function_zhou, fun.function_zakharov, fun.function_styblinski_tang]
    dimensions = [1, 2, 3]
    n_samples = [2, 4, 8, 16, 32]
    num_runs_per_set = 20
    
    # CARGAR CHECKPOINT SI EXISTE
    all_metrics_dim, progress_tracker, loaded_config = load_checkpoint()
    
    # Inicializar experiment_config_data (usar loaded si existe)
    experiment_config_data = loaded_config if loaded_config is not None else {
        "Métricas": ["mae", "mse", "time"],
        "Dimensiones": dimensions,
        "Repeticiones": num_runs_per_set,
        "Funciones": [func.__name__ for func in functions],
        "Tamaño del dataset 1D": n_samples
    }
    
    wandb.init(
        project="PySESM_experiments",
        config={
            "functions": [func.__name__ for func in functions],
            "function_limits": function_limits,
            "dimensions": dimensions,
            "n_samples_1D": n_samples,
            "num_runs_per_set": num_runs_per_set,
            "resume": len(progress_tracker) > 0
        }
    )
    
    for function in functions:
        function_name = function.__name__
        
        # Inicializar all_metrics_dim para esta función si no existe
        if function_name not in all_metrics_dim:
            all_metrics_dim[function_name] = {}
        
        for dim in dimensions:
            # Inicializar configuraciones de modelos
            svr_config = {"kernel": 'rbf', "C": 0.01, "gamma": 'auto', "epsilon": 0.1}
            nn_config = {"epochs": 500, "lr": 0.01, "hidden_dim": 16, "input_d": dim}
            pf_config = {"order": 3, "alpha": 0.01, "include_bias": True, "max_iter": 10000}
            
            sparse_coding_config = ISTAConfig(
                epochs=110,
                alpha=0.1,
                lambd=5e-4,
                step_size_method=StepSizeMethod.FROBENIUS,
                power_iterations=10,
                n_functions=8**dim,
                criterion=torch.nn.MSELoss(),
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )

            dict_config = GaussianDictConfig(
                epochs=300,
                alpha=0.001107722632753506,
                criterion=torch.nn.MSELoss(),
                optimizer_factory=lambda params, lr: torch.optim.AdamW(params, lr=lr),
                mu_epochs=1, 
                rho_epochs=1, 
                split_mu_rho=False,
                eig_range=[0.05, 0.2],
                regularization_func=GaussianDictLayer.electrostatic_regularization,
                regularization_gamma=2e-8,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )

            [x1lim, x2lim] = function_limits[function_name]

            strategy = KDTreeStrategy
            strategyConfig = KDTreeStrategyConfig(
                maxNodeSize=5,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                data_wrapper=SESMData
            )
            partition_config = AdaptativePartitionConfig(
                overlap_ratio=0.1,
                partition_strategy=strategy,
                strategy_config=strategyConfig
            )

            ssesm_config = SSESMConfig(
                n_features=dim, 
                model_epochs=50,
                sparse_coding_config=sparse_coding_config,
                dict_config=dict_config,
                partition_config=partition_config,
                log_interval=25,
                permutation_times=1,
                seed=45,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )

            # Inicializar all_metrics para esta dimensión si no existe
            if dim not in all_metrics_dim[function_name]:
                all_metrics_dim[function_name][dim] = defaultdict(list)
            
            all_metrics = all_metrics_dim[function_name][dim]

            # Calcular tamaños de datasets por dimensión
            n_samples_dim = [int(n**dim) for n in n_samples]
            
            for n_idx, n in enumerate(n_samples_dim):
                # Inicializar chunk_metrics para este n si no existen las listas
                # Necesitamos verificar si ya hay datos para este n_idx
                needs_initialization = False
                if len(all_metrics) == 0:
                    needs_initialization = True
                else:
                    # Verificar si alguna métrica tiene datos para este n_idx
                    sample_key = list(all_metrics.keys())[0]
                    if len(all_metrics[sample_key]) <= n_idx:
                        needs_initialization = True
                
                if needs_initialization:
                    # Agregar listas vacías para este n a todas las métricas existentes
                    for key in ['MSE_SESM', 'MSE_SVR', 'MSE_NN', 'MSE_PF',
                                'MAE_SESM', 'MAE_SVR', 'MAE_NN', 'MAE_PF',
                                'TIME_SESM', 'TIME_SVR', 'TIME_NN', 'TIME_PF']:
                        if len(all_metrics[key]) <= n_idx:
                            all_metrics[key].append([])
                
                for j in range(num_runs_per_set):
                    # VERIFICAR SI ESTA COMBINACIÓN YA FUE COMPLETADA
                    if is_combination_done(progress_tracker, function_name, dim, n, j):
                        logger.debug("Saltando: %s, dim=%d, n=%d, run=%d (ya completado)",function_name,dim,n,j+1)
                        continue
                    
                    logger.debug(r"""
                    =================================================================================
                                                ,___,      Repetition:      %d/%d
                            0/     (\(\         (O.o)      Dataset size:    %d                      
                            <|      (-.-)        /),,)      Dimension:       %d             
                            / \     o_(")(")      " "       Function:        %s    
                    =================================================================================
                    """,j+1,num_runs_per_set,n,dim,function_name)
                    
                    torch.manual_seed(j)
                    
                    # Generar dataset
                    dataset_config = {
                        "n_samples": n,
                        "n_dimensions": dim, 
                        "function": function,
                        "limits": function_limits[function_name]
                    }
                    train_data, _, _, test_data, _, _ = generate_custom_nd_function_dataset(**dataset_config)

                    # Entrenar y testear
                    experiment = EXPERIMENT(svr_config, nn_config, ssesm_config, pf_config)
                    experiment.setup_dataset(train_data, test_data)
                    experiment.train_all()
                    logger.debug("Debug, device: %s", experiment.SESM_model.partition_manager.kdtree.device)
                    experiment.test_all()
                    metrics = experiment.metrics

                    # Guardar métricas de este run en all_metrics
                    for key, value in metrics.items():
                        all_metrics[key][n_idx].append(value)
                    
                    # MARCAR ESTA COMBINACIÓN COMO COMPLETADA
                    mark_combination_done(progress_tracker, function_name, dim, n, j)
                    
                    # GUARDAR CHECKPOINT DESPUÉS DE CADA RUN
                    save_checkpoint(all_metrics_dim, progress_tracker, experiment_config_data)
                    del experiment, train_data, test_data
                    torch.cuda.empty_cache()  # seguro aunque estés en CPU
                    gc.collect()
        # Guardar resultados finales de esta función (compatibilidad con código original)
        joblib.dump(all_metrics_dim[function_name], "./plots/metrics/metrics_"+str(function_name)+".joblib")
        joblib.dump(n_samples, "./plots/metrics/n_samples.joblib")
    
    # GUARDAR CONFIGURACIONES DEL EXPERIMENTO
    joblib.dump(experiment_config_data, "./plots/config/config_experiment.joblib")
    
    # Guardar configs de modelos (usando el último experiment)
    if 'experiment' in locals():
        experiment.save_configs()
    
    # Limpiar checkpoint al finalizar exitosamente
    checkpoint_path = "./checkpoints/checkpoint.joblib"
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        logger.debug("Experimento completado exitosamente. Checkpoint eliminado.")
    
    wandb.finish()
    
    logger.debug("Experimento completado. Métricas para boxplots listas.")


if __name__ == "__main__":
    main()
