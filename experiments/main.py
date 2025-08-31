import os
import torch
import numpy as np
import wandb
from prepare_all import EXPERIMENT

from pysesm.utils_dataset.generate_dataset import generate_custom_function_dataset
from LossWrappers import KLDivLossWrapper, JensenShannonLossWrapper, CrossEntropyLossWrapper

from pysesm.models.SSESM import SSESMConfig
from pysesm.sparse_coding import ISTALayer, ISTAConfig, StepSizeMethod
from pysesm.dictionaries import GaussianDictLayer, GaussianDictConfig
from pysesm.blocks.UniformPartitionManager import UniformPartitionConfig
from pysesm.utils.loggers import setup_logger
from pysesm.utils_dataset.generate_dataset import generate_gaussian_dataset
from pysesm.utils.plot_and_save_stats import plot_surface
from pysesm.utils.metric_loggers import *
from pysesm.enums.DeviceTargetEnum import DeviceTarget
from pysesm.device_manager.DeviceManager import DeviceManager

def main():
    """
    Script de experimento para comparar el rendimiento de SVR y Redes Neuronales 
    en una tarea de regresión. Genera un conjunto de datos personalizado,
    entrena ambos modelos y registra las métricas de rendimiento en W&B.
    """

    # 1. Configuración del Experimento y del Dataset
    # ----------------------------------------------------
    def custom_function(x, y):
        """Función personalizada para generar datos 2D."""
        pi = np.pi
        return torch.sin(pi * x) / (pi * x) - torch.sin(pi * y) / (pi * y)

    # Parámetros del experimento
    n_samples=500
    dataset_config = {
        "n_samples": n_samples,
        "function": custom_function,
        "mesh_divisions": 70
    }

    svr_config = {
        "kernel": 'rbf',
        "C": 0.1,
        "gamma": 'auto',
        "epsilon": 0.1
    }

    nn_config = {
        "epochs": 500,
        "lr": 0.01,
        "hidden_dim": 16
    }

    sparse_coding_config = ISTAConfig(
        epochs=200,
        alpha=0.10,
        lambd=0.00001,
        step_size_method=StepSizeMethod.FROBENIUS,  # POWER_ITERATION,
        power_iterations=10,
        n_functions=10,
        criterion=torch.nn.MSELoss()
    )

    dict_config = GaussianDictConfig(
        epochs = 4,
        alpha = 0.01,
        # criterion = torch.nn.MSELoss(),
        # criterion = KLDivLossWrapper(),
        criterion = JensenShannonLossWrapper(),
        optimizer_factory = lambda params, lr: torch.optim.SGD(params, lr=lr, momentum=0.1),
        mu_epochs = 10,
        rho_epochs = 10,
        split_mu_rho = True,
        eig_range = [0.05, 0.2],
        mu_range = [-2.0, 2.0],
    )

    partition_config = UniformPartitionConfig(
        T=1,
        initial_bounds = torch.tensor([[-2, -2], [2, 2]], dtype=torch.float32),
        activity_threshold=0,
        overlap_ratio=0.25
    )

    ssesm_config = SSESMConfig(
        n_features = 2,
        model_epochs = 100,
        sparse_coding_config = sparse_coding_config,
        dict_config = dict_config,
        partition_config = partition_config,
        log_interval=100,
        permutation_times=1
    )

    experiment1 = {
        "config": ssesm_config,
        "hyp_set": 1,
        "n_samples": n_samples,
        "seed": 45,
        "iter": 0,
        "device_map": {
            DeviceTarget.GLOBAL: "cpu",               # Dispositivo global por defecto
            DeviceTarget.SPARSE_CODING_LAYER: "cpu",  # ISTA en GPU 0
            DeviceTarget.DICTIONARY_LAYER: "cpu",     # Dictionary en CPU
            DeviceTarget.PARTITION_MANAGER: "cpu"     # Partition Manager en CPU
        }
    }

    num_runs = 10 # Aumentar el número de corridas para un análisis estadístico más robusto

    wandb.init(
        project="PySESM_experiments",
        config={
            "dataset_config": dataset_config,
            "svr_config": svr_config,
            "nn_config": nn_config,
            "SESM_config": ssesm_config,
            "num_runs": num_runs
        }
    )

    # 3. Ciclo de Entrenamiento y Recolección de Métricas
    # ----------------------------------------------------
    all_metrics = {
        "NN_MAE": [], "NN_MSE": [],
        "SVR_MAE": [], "SVR_MSE": [],
        "SESM_MAE": [], "SESM_MSE":[]
    }

    for i in range(num_runs):
        print(f"--- Corriendo experimento {i + 1}/{num_runs} ---")

        experiment = EXPERIMENT(svr_config, nn_config, experiment1)
        
        # Generar un nuevo dataset en cada corrida para validar la robustez
        train_data, _, _, test_data, _, _ = generate_custom_function_dataset(**dataset_config)
        
        # Entrenar y evaluar los modelos
        experiment.train_all(train_data, test_data)
        
        # El flag de plot solo se activa en la última iteración
        plot_flag = (i == num_runs - 1)

        metrics = experiment.test_all(train_data, test_data, plot_flag)
        
        # Almacenar las métricas en un diccionario para un análisis posterior
        for key in all_metrics.keys():
            all_metrics[key].append(metrics[key])

    # 4. Análisis y Registro de Resultados
    # ----------------------------------------------------
    # Calcular promedios y desviaciones estándar
    summary_metrics = {}
    for key, values in all_metrics.items():
        summary_metrics[f"mu_{key}"] = np.mean(values)
        summary_metrics[f"std_{key}"] = np.std(values)

    # Registrar las métricas de resumen en W&B
    wandb.log(summary_metrics)
    
    # Finalizar el experimento
    wandb.finish()
    print("Experimento completado. Los resultados han sido registrados en Weights & Biases.")


if __name__ == "__main__":
    main()