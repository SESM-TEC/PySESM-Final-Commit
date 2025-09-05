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



    svr_config = {"kernel": 'rbf', "C": 0.1, "gamma": 'auto', "epsilon": 0.1}
    nn_config = {"epochs": 500, "lr": 0.01, "hidden_dim": 16}
    pf_config = {"order": 5, "alpha": 0.01}
    
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

    #TODO: el diccionario experiment1 está pidiendo n_samples, pero n_samples es un valor que variaa a lo largo del experimento
    n_samples = 16  
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



     # 3) Estructura de métricas: para CADA métrica -> lista de tamaño num_chunks,
    #     y cada elemento es una lista donde iremos agregando los valores de CADA run.
    all_metrics = { 
        "NN_MAE": [], "NN_MSE": [], 
        "SVR_MAE": [], "SVR_MSE": [], 
        "SESM_MAE": [], "SESM_MSE": [],
        "PF_MAE": [], "PF_MSE": []
    }
    num_runs_per_set = 20  # Define cuántos entrenamientos por chunk quieres
    n_samples = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    
    # 2) Inicializar Weights & Biases una sola vez (registraremos por run y chunk)
    wandb.init(
        project="PySESM_experiments",
        config={
            "svr_config": svr_config,
            "nn_config": nn_config,
            "SESM_config": ssesm_config,
            "PF_config": pf_config,
            "num_runs_per_set": num_runs_per_set
        }
    )

    for n in n_samples: # los valores del vector n_samples
        print(f"--- Entrenando con {n} muestras ---")

        # Listas temporales para almacenar las métricas de este chunk
        chunk_metrics = { 
            "NN_MAE": [], "NN_MSE": [],
            "SVR_MAE": [], "SVR_MSE": [],
            "SESM_MAE": [], "SESM_MSE": [],
            "PF_MAE": [], "PF_MSE": []
        }

        # Bucle para correr múltiples experimentos en el mismo chunk
        for j in range(num_runs_per_set):
            print(f"--- Entrenamiento numero {j} con {n} muestras ---")

            #Se genera un nuevo dataset con n muestras
            dataset_config = { "n_samples": n, "function": custom_function}
            train_data, _, _, test_data, _, _ = generate_custom_function_dataset(**dataset_config)

            
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






    # 5) Boxplots con la misma API que tenías (lista-de-listas por métrica)
    #    Usamos el último 'experiment' creado; si prefieres, llama al método desde otro objeto.
    experiment.plot_caja_bigote(all_metrics, n_samples)

    wandb.finish()
    print("Experimento completado. Métricas para boxplots listas.")



if __name__ == "__main__":
    main()