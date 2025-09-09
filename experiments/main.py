import os
import torch
import numpy as np
import joblib
import wandb
from prepare_all import EXPERIMENT

from pysesm.utils_dataset.generate_dataset import generate_custom_function_dataset, generate_custom_nd_function_dataset
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
    def sinc_function(X):
        x, y = X[:, 0], X[:, 1]
        pi = np.pi
        return torch.sin(pi * x) / (pi * x) - torch.sin(pi * y) / (pi * y)
    
    def sinc_3d_function(X):
        features=[X[:, feature] for feature in range(X.size(1))]
        pi = np.pi
        out=0
        for idx, feature in enumerate(features):
            out=out+((-1)^idx)*torch.sin(pi*feature)/(pi*feature)

        return out
    
    n_dimensions= 3 
    all_metrics_dim={}
    all_times_dim={}

    for dim in range(2,n_dimensions+1):
        num_runs_per_set = 20


        svr_config = {"kernel": 'rbf', "C": 0.1, "gamma": 'auto', "epsilon": 0.1}
        nn_config = {"epochs": 500, "lr": 0.01, "hidden_dim": 16, "input_d":dim}
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
            T=2,
            initial_bounds=torch.tensor([[-2 for i in range(dim)], [2 for i in range(dim)]], dtype=torch.float32),
            activity_threshold=0, overlap_ratio=0.1
        )

        ssesm_config = SSESMConfig(
            n_features= dim, model_epochs=200,
            sparse_coding_config=sparse_coding_config,
            dict_config=dict_config, partition_config=partition_config,
            log_interval=100, permutation_times=1
        )

        #TODO: el diccionario experiment1 está pidiendo n_samples, pero n_samples es un valor que varia a lo largo del experimento
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



        # Definir las métricas y tiempos una sola vez
        METRICS = ["NN_MAE", "NN_MSE", "SVR_MAE", "SVR_MSE", 
                "SESM_MAE", "SESM_MSE", "PF_MAE", "PF_MSE"]

        TIMES = ["svr_time", "nn_time", "sesm_time", "pf_time"]


        def init_dict(keys):
            """Inicializa un diccionario con listas vacías por clave."""
            return {key: [] for key in keys}


        # 1) Diccionarios principales
        all_metrics = init_dict(METRICS)
        all_times   = init_dict(TIMES)

        
        # 2) Inicializar Weights & Biases una sola vez
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
        
        n_samples = [8, 16, 32, 64, 128, 256, 512, 1024]    
        for n in n_samples:

            # Diccionarios temporales para este chunk
            chunk_metrics = init_dict(METRICS)
            chunk_times   = init_dict(TIMES)

            for j in range(num_runs_per_set):
                print(f"--- Entrenamiento número {j} con {n} muestras ---")

                # Generar dataset
                dataset_config = {"n_samples": n,"n_dimensions":dim, "function": sinc_3d_function}
                train_data, _, _, test_data, _, _ = generate_custom_nd_function_dataset(**dataset_config)

                # Crear experimento y entrenar
                experiment = EXPERIMENT(svr_config, nn_config, experiment1, pf_config)
                times = experiment.train_all(train_data, test_data)
                metrics = experiment.test_all(train_data, test_data, plot_flag=False)

                # Guardar resultados
                for key in METRICS:
                    chunk_metrics[key].append(metrics[key])
                for key in TIMES:
                    chunk_times[key].append(times[key])

            # Guardar resultados finales del chunk
            for key in METRICS:
                all_metrics[key].append(chunk_metrics[key])
            for key in TIMES:
                all_times[key].append(chunk_times[key])
        all_metrics_dim[dim]=all_metrics
        all_times_dim[dim]=all_times

    joblib.dump(all_metrics_dim, "./plots/all_metrics.joblib")
    joblib.dump(all_times_dim, "./plots/all_times.joblib")
    joblib.dump(n_samples, "./plots/n_samples.joblib")

    wandb.finish()
    print("Experimento completado. Métricas para boxplots listas.")



if __name__ == "__main__":
    main()