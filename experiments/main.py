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

    num_runs = 1

    # 1) Crear chunks con tamaño creciente logarítmicamente (no solapados)
    train_chunks = []
    start = 0
    i = 1
    while start < len(X_full):
        # evita tamaños 0 si n_samples es pequeño
        chunk_size = max(1, int(np.log(i + 1) * n_samples))  # escala log
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

    # 2) Inicializar Weights & Biases una sola vez (registraremos por run y chunk)
    wandb.init(
        project="PySESM_experiments",
        config={
            "dataset_config": dataset_config,
            "svr_config": svr_config,
            "nn_config": nn_config,
            "SESM_config": ssesm_config,
            "num_runs": num_runs,  # número de corridas completas
            "num_chunks": len(train_chunks),
        },
    )

    # 3) Estructura de métricas: para CADA métrica -> lista de tamaño num_chunks,
    #     y cada elemento es una lista donde iremos agregando los valores de CADA run.
    metric_keys = [
        "NN_MAE", "NN_MSE",
        "SVR_MAE", "SVR_MSE",
        "SESM_MAE", "SESM_MSE",
        "PCE_MAE", "PCE_MSE"
    ]
    all_metrics = {k: [[] for _ in range(len(train_chunks))] for k in metric_keys}

    # 4) Bucle de corridas completas
    for run in range(num_runs):
        print(f"\n=== Run {run+1}/{num_runs} ===")

        # Entrenamiento y test por chunk (UNA vez por chunk y por run)
        for ci, train_data in enumerate(train_chunks):
            print(f"--- Run {run+1}, Chunk {ci+1}/{len(train_chunks)} con {len(train_data['X'])} muestras ---")

            experiment = EXPERIMENT(svr_config, nn_config, experiment1, pf_config)

            # Entrenar con el chunk
            experiment.train_all(train_data, test_data)

            # Testear con el mismo chunk
            metrics = experiment.test_all(train_data, test_data, plot_flag=False)

            # Acumular: por métrica -> en la lista del chunk correspondiente
            for k in metric_keys:
                all_metrics[k][ci].append(metrics[k])

            # Log en wandb con etiquetas de run y chunk
            wandb.log({f"{k}_run{run+1}_chunk{ci+1}": metrics[k] for k in metric_keys})

    # 5) Boxplots con la misma API que tenías (lista-de-listas por métrica)
    #    Usamos el último 'experiment' creado; si prefieres, llama al método desde otro objeto.
    experiment.plot_caja_bigote(all_metrics)

    wandb.finish()
    print("Experimento completado. Métricas para boxplots listas.")



if __name__ == "__main__":
    main()