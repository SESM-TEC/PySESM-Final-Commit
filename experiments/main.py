import os
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
from pysesm.sparse_coding import ISTALayer, ISTAConfig, StepSizeMethod
from pysesm.dictionaries import GaussianDictLayer, GaussianDictConfig
from pysesm.blocks.UniformPartitionManager import UniformPartitionConfig
from pysesm.utils.metric_loggers import *
from pysesm.enums.DeviceTargetEnum import DeviceTarget

def main():
    """
    Script para correr múltiples experimentos en chunks de tamaño n_samples.
    Cada chunk se usa para entrenar y luego testear, guardando métricas por corrida.
    """
    # CONFIGURACIONES DEL EXPERIMENTO
    function_limits = {
        "zakharov_function" : [-10, 10],
        "styblinski_tang_function": [-5, 5],
        "zhou_function": [0, 1]
    }
    functions=[fun.zakharov_function, fun.styblinski_tang_function, fun.zhou_function]
    dimensions= [1,2] # CAMBIAR A [1, 2, 3, 4] DIMENSIONES
    n_samples = [4, 8, 16, 32, 64]  # CAMBIAR A [4, 8, 16, 32, 64] #TODO: quizas lineal funcionaria mejor
    num_runs_per_set = 10 # CAMBIAR A 50 




    all_metrics_dim={}
    all_times_dim={}
    for function in functions:
        for dim in dimensions:

            svr_config = {"kernel": 'rbf', "C": 0.01, "gamma": 'auto', "epsilon": 0.1}
            nn_config = {"epochs": 500, "lr": 0.01, "hidden_dim": 16, "input_d":dim}
            pf_config = {"order": 3, "alpha": 0.01, "include_bias": True, "max_iter": 10000}
            
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

            [x1lim, x2lim] = function_limits[function.__name__]
            partition_config = UniformPartitionConfig(
                T=1,
                initial_bounds=torch.tensor([[x1lim for i in range(dim)], [x2lim for i in range(dim)]], dtype=torch.float32),
                activity_threshold=0, overlap_ratio=0.1
            )

            ssesm_config = SSESMConfig(
                n_features= dim, model_epochs=100,
                sparse_coding_config=sparse_coding_config,
                dict_config=dict_config, partition_config=partition_config,
                log_interval=100, permutation_times=1
            )

            experiment1 = {
                "config": ssesm_config, "hyp_set": 1, "n_samples": 0,
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
            
            # Se recalcula el tamaño del dataset en cada dimension
            n_samples_dim = [int(n**dim) for n in n_samples]
            for n in n_samples_dim:

                # Diccionarios temporales para este chunk
                chunk_metrics = defaultdict(list)
                chunk_times   = defaultdict(list)

                for j in range(num_runs_per_set):
                    
                    print(rf"""
                    =================================================================================
                                              ,___,      Repetition:      {j+1}/{num_runs_per_set}
                          0/     (\(\         (O.o)      Dataset size:    {n}                      
                         <|      (-.-)        /),,)      Dimension:       {dim}             
                         / \     o_(")(")      " "       Function:        {function.__name__}    
                    =================================================================================
                    """)
                    
                
                    # Generar dataset con n muestras, de dim dimensiones con la funcion function
                    dataset_config = {
                        "n_samples": n,
                        "n_dimensions":dim, 
                        "function": function,
                        "limits": function_limits[function.__name__]}
                    train_data, _, _, test_data, _, _ = generate_custom_nd_function_dataset(**dataset_config)

                    # Crear experimento y entrenar con el metodo train_all y testear con test_all
                    # Se entrenan y testean 4 modelos j veces 
                    experiment = EXPERIMENT(svr_config, nn_config, experiment1, pf_config)
                    experiment.setup_dataset(train_data, test_data)
                    times = experiment.train_all()
                    metrics = experiment.test_all()

                    # Guardar mse y mae de cada modelo j veces
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
        
        joblib.dump(all_metrics_dim, "./plots/all_metrics_"+str(function.__name__)+".joblib")
        joblib.dump(all_times_dim, "./plots/all_times_"+str(function.__name__)+".joblib")
        joblib.dump(n_samples, "./plots/n_samples.joblib")

    wandb.finish()
    print("Experimento completado. Métricas para boxplots listas.")



if __name__ == "__main__":
    main()