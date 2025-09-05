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
    Script para correr múltiples experimentos en chunks de tamaño n_samples.
    Cada chunk se usa para entrenar y luego testear, guardando métricas por corrida.
    """

    # 1. Configuración del experimento y dataset
    def custom_function(x, y):
        pi = np.pi
        return torch.sin(pi * x) / (pi * x) - torch.sin(pi * y) / (pi * y)

    n_samples = 10  # tamaño de cada chunk
    total_samples = 100  # total de muestras a generar

    dataset_config = {
        "n_samples": total_samples,
        "function": custom_function,
        "mesh_divisions": 70
    }

    svr_config = {"kernel": 'rbf', "C": 0.1, "gamma": 'auto', "epsilon": 0.1}
    nn_config = {"epochs": 500, "lr": 0.01, "hidden_dim": 16}

    sparse_coding_config = ISTAConfig(
        epochs=200, alpha=0.1, lambd=0.00001,
        step_size_method=StepSizeMethod.FROBENIUS,
        power_iterations=10, n_functions=10,
        criterion=torch.nn.MSELoss()
    )

    dict_config = GaussianDictConfig(
        epochs=40, alpha=0.01,
        criterion=JensenShannonLossWrapper(),
        optimizer_factory=lambda params, lr: torch.optim.SGD(params, lr=lr, momentum=0.1),
        mu_epochs=10, rho_epochs=10, split_mu_rho=True,
        eig_range=[0.05, 0.2], mu_range=[-2.0, 2.0],
    )

    partition_config = UniformPartitionConfig(
        T=1,
        initial_bounds=torch.tensor([[-2, -2], [2, 2]], dtype=torch.float32),
        activity_threshold=0, overlap_ratio=0.1
    )

    ssesm_config = SSESMConfig(
        n_features=2, model_epochs=200,
        sparse_coding_config=sparse_coding_config,
        dict_config=dict_config, partition_config=partition_config,
        log_interval=100, permutation_times=1
    )

    experiment1 = {
        "config": ssesm_config, "hyp_set": 1, "n_samples": n_samples,
        "seed": 45, "iter": 0,
        "device_map": {
            DeviceTarget.GLOBAL: "cpu",
            DeviceTarget.SPARSE_CODING_LAYER: "cpu",
            DeviceTarget.DICTIONARY_LAYER: "cpu",
            DeviceTarget.PARTITION_MANAGER: "cpu"
        }
    }

    pf_config = {"order": 5, "alpha": 0.01}

    # Generar dataset completo una vez
    # Generar dataset completo una vez
    full_train, _, _, test_data, _, _ = generate_custom_function_dataset(**dataset_config)

    # full_train is a dict with keys "X","Y","Z"
    X_full = full_train["X"]
    Y_full = full_train["Y"]
    Z_full = full_train["Z"]

    # Crear chunks con tamaño creciente logarítmicamente
    train_chunks = []
    start = 0
    i = 1
    while start < len(X_full):
        chunk_size = int(np.log(i + 1) * n_samples)  # escala con log
        end = min(start + chunk_size, len(X_full))

        chunk = {
            "X": X_full[start:end],
            "Y": Y_full[start:end],
            "Z": Z_full[start:end],
        }
        train_chunks.append(chunk)

        start = end
        i += 1

    print(f"Created {len(train_chunks)} chunks — sizes:", [len(c["X"]) for c in train_chunks])

    wandb.init(
        project="PySESM_experiments",
        config={
            "dataset_config": dataset_config,
            "svr_config": svr_config,
            "nn_config": nn_config,
            "SESM_config": ssesm_config,
            "num_runs": len(train_chunks),
        },
    )

    # Diccionario de métricas por chunk
    all_metrics = { 
        "NN_MAE": [], 
        "NN_MSE": [], 
        "SVR_MAE": [], 
        "SVR_MSE": [], 
        "SESM_MAE": [], 
        "SESM_MSE": [],
        "PCE_MAE": [],
        "PCE_MSE": []
    }
    num_runs_per_chunk = 1  # Define cuántos entrenamientos por chunk quieres

    # Entrenamiento y test por chunk (cada chunk una sola vez)
    for i, train_data in enumerate(train_chunks):
        print(f"Created {len(train_chunks)} chunks — sizes:", [len(c["X"]) for c in train_chunks])

        # Listas temporales para almacenar las métricas de este chunk
        chunk_metrics = { 
            "NN_MAE": [],
            "NN_MSE": [],
            "SVR_MAE": [],
            "SVR_MSE": [],
            "SESM_MAE": [],
            "SESM_MSE": [],
            "PCE_MAE": [],
            "PCE_MSE": []
        }

        # Bucle para correr múltiples experimentos en el mismo chunk
        print(f"--- Entrenando y testeando run en el chunk {i + 1} ---")
        
        experiment = EXPERIMENT(svr_config, nn_config, experiment1, pf_config)

        # 1) Entrenar con el chunk
        experiment.train_all(train_data, test_data)

        # 2) Testear con el mismo chunk
        metrics = experiment.test_all(train_data, test_data, plot_flag=False)

        # Guardar métricas de esta corrida en las listas temporales
        for key in chunk_metrics.keys():
            chunk_metrics[key].append(metrics[key])

        # Después de todas las corridas de este chunk, añadir los resultados a las listas principales
        for key in all_metrics.keys():
            all_metrics[key].append(chunk_metrics[key])

        # Registrar métricas promedio del chunk en wandb (opcional)
        # avg_metrics = {key: sum(chunk_metrics[key])/len(chunk_metrics[key]) for key in chunk_metrics.keys()}
        # wandb.log({f"{key}_avg_chunk{i+1}": avg_metrics[key] for key in avg_metrics.keys()})
    experiment.plot_caja_bigote(all_metrics)

wandb.finish()
print("Experimento completado. Métricas para boxplots listas.")

if __name__ == "__main__":
    main()