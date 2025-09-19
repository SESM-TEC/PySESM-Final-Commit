import os
import torch
import numpy as np
import joblib
import wandb
from prepare_all import EXPERIMENT
from collections import defaultdict

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

    # Funcion con más oscilaciones
    def zhou_function(X: torch.Tensor) -> torch.Tensor:
        """
        Zhou (1998) function.

        Args:
            X (torch.Tensor): Input tensor of shape (n_samples, n_dimensions).
                            Each row is a point in the search space (values typically in [0,1]).

        Returns:
            torch.Tensor: Function values of shape (n_samples,).
        """
        d = X.shape[1]

        # Shift and scale
        X_a = 10 * (X - 1.0/3.0)
        X_b = 10 * (X - 2.0/3.0)

        # Norms squared
        norm_a2 = torch.sum(X_a**2, dim=1)
        norm_b2 = torch.sum(X_b**2, dim=1)

        # Gaussian components
        coeff = (2 * torch.pi) ** (-d / 2)
        phi1 = coeff * torch.exp(-0.5 * norm_a2)
        phi2 = coeff * torch.exp(-0.5 * norm_b2)

        # Final result
        return (10.0**d) / 2.0 * (phi1 + phi2)
    
    #Funcion con algunos valles o montañas "suaves"
    def rosenbrock_rescaled_function(X: torch.Tensor) -> torch.Tensor:
        """
        Rescaled Rosenbrock function.

        Args:
            X (torch.Tensor): Input tensor of shape (n_samples, 4).
                            Each row is a point in the search space, with values in [0,1].

        Returns:
            torch.Tensor: Function values of shape (n_samples,).
        """
        # Rescale inputs: [0,1] -> [-5,10]
        Xbar = 15 * X - 5  # shape (n_samples, 4)

        # Rosenbrock sum
        sum_terms = torch.sum(
            100.0 * (Xbar[:, 1:] - Xbar[:, :-1]**2)**2 + (1 - Xbar[:, :-1])**2,
            dim=1
        )

        # Rescale output
        return (sum_terms - 3.827e5) / 3.755e5

    #Funcion tipo parábola
    def zakharov_function(X: torch.Tensor) -> torch.Tensor:
        """
        Zakharov function.

        Args:
            X (torch.Tensor): Input tensor of shape (n_samples, n_dimensions).
                            Each row is a point in the search space.

        Returns:
            torch.Tensor: Function values of shape (n_samples,).
        """
        # indices for 1..d (broadcasted to match X)
        d = X.shape[1]
        ii = torch.arange(1, d + 1, dtype=X.dtype, device=X.device).unsqueeze(0)

        # sum1 = sum(x_i^2)
        sum1 = torch.sum(X**2, dim=1)

        # sum2 = sum(0.5 * i * x_i)
        sum2 = torch.sum(0.5 * ii * X, dim=1)

        # final function value
        return sum1 + sum2**2 + sum2**4

    
    def plane_function(X):
        """Esta funcion recibe un tensor 2d y retorna la suma de las columnas."""
        return torch.sum(X, dim=1) 


    functions=[zakharov_function, rosenbrock_rescaled_function, zhou_function]
    n_dimensions= 3
    num_runs_per_set = 2
    n_samples = [64, 256]  # Número de muestras por chunk (debe ser una lista de enteros)

    all_metrics_dim={}
    all_times_dim={}
  
    for function in functions:
        for dim in range(2,n_dimensions+1):

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
                T=1,
                initial_bounds=torch.tensor([[-2 for i in range(dim)], [2 for i in range(dim)]], dtype=torch.float32),
                activity_threshold=0, overlap_ratio=0.1
            )

            ssesm_config = SSESMConfig(
                n_features= dim, model_epochs=100,
                sparse_coding_config=sparse_coding_config,
                dict_config=dict_config, partition_config=partition_config,
                log_interval=100, permutation_times=1
            )

            #TODO: el diccionario experiment1 está pidiendo n_samples, pero n_samples es un valor que varia a lo largo del experimento
            sesm_n_samples = 4
            experiment1 = {
                "config": ssesm_config, "hyp_set": 1, "n_samples": sesm_n_samples,
                "seed": 45, "iter": 0,
                "device_map": {
                    DeviceTarget.GLOBAL: "cpu",
                    DeviceTarget.SPARSE_CODING_LAYER: "cpu",
                    DeviceTarget.DICTIONARY_LAYER: "cpu",
                    DeviceTarget.PARTITION_MANAGER: "cpu"
                }
            }



            # 1) Diccionarios principales
            all_metrics = defaultdict(list)
            all_times   = defaultdict(list)

            
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
            

            n_samples_dim = [int(n**(dim/2)) for n in n_samples]

            for n in n_samples_dim:

                # Diccionarios temporales para este chunk
                chunk_metrics = defaultdict(list)
                chunk_times   = defaultdict(list)

                for j in range(num_runs_per_set):
                    print(f"\n\n --- Entrenamiento número {j} con {n} muestras en {dim}D de la función {function.__name__} ---\n\n")

                    # Generar dataset
                    dataset_config = {"n_samples": n,"n_dimensions":dim, "function": function}
                    train_data, _, _, test_data, _, _ = generate_custom_nd_function_dataset(**dataset_config)

                    # Crear experimento y entrenar
                    experiment = EXPERIMENT(svr_config, nn_config, experiment1, pf_config)
                    times = experiment.train_all(train_data, test_data)
                    metrics = experiment.test_all(train_data, test_data, plot_flag=False)

                    # Guardar resultados
                    for key, value in metrics.items(): 
                        chunk_metrics[key].append(value)
                    for key, value in times.items():
                        chunk_times[key].append(value)
    
                # Guardar resultados finales del chunk
                for key, value in chunk_metrics.items():
                    all_metrics[key].append(value)
                for key, value in chunk_times.items():
                    all_times[key].append(value)

            all_metrics_dim[dim]=all_metrics
            all_times_dim[dim]=all_times

        joblib.dump(all_metrics_dim, "./plots/all_metrics"+str(function.__name__)+".joblib")
        joblib.dump(all_times_dim, "./plots/all_times"+str(function.__name__)+".joblib")
        joblib.dump(n_samples, "./plots/n_samples.joblib")

    wandb.finish()
    print("Experimento completado. Métricas para boxplots listas.")



if __name__ == "__main__":
    main()