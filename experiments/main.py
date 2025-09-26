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
from pysesm.sparse_coding import ISTAConfig, StepSizeMethod
from pysesm.dictionaries import GaussianDictConfig, GaussianDictLayer
from pysesm.blocks.UniformPartitionManager import UniformPartitionConfig
from pysesm.utils.metric_loggers import *

def main():
    """
    Script para correr múltiples experimentos en chunks de tamaño n_samples.
    Cada chunk se usa para entrenar y luego testear, guardando métricas por corrida.
    """

    
    # CONFIGURACIONES DEL EXPERIMENTO
    function_limits = {
        "function_zakharov" : [-10, 10],
        "function_styblinski_tang": [-5, 5],
        "function_zhou": [0, 1]
    }
    functions=[fun.function_zhou, fun.function_zakharov, fun.function_styblinski_tang]
    dimensions= [1, 2] # CAMBIAR A [1, 2, 3, 4] DIMENSIONES
    n_samples = [2, 4, 8, 16, 32]  # CAMBIAR A [4, 8, 16, 32, 64] #TODO: quizas lineal funcionaria mejor
    num_runs_per_set = 15 # CAMBIAR A 50 
    
    wandb.init(
        project="PySESM_experiments",
        config={
            "functions": [func.__name__ for func in functions],
            "function_limits": function_limits,
            "dimensions": dimensions,
            "n_samples_1D": n_samples,
            "num_runs_per_set": num_runs_per_set
        }
    )
    

    all_metrics_dim={}
    for function in functions:
        for dim in dimensions:

            svr_config = {"kernel": 'rbf', "C": 0.01, "gamma": 'auto', "epsilon": 0.1}
            nn_config = {"epochs": 500, "lr": 0.01, "hidden_dim": 16, "input_d":dim}
            pf_config = {"order": 3, "alpha": 0.01, "include_bias": True, "max_iter": 10000}
            
            sparse_coding_config = ISTAConfig(
                epochs=150,
                alpha=0.1,
                lambd=1e-3,
                step_size_method=StepSizeMethod.FROBENIUS,
                power_iterations=10,
                n_functions= 8**dim,
                criterion=torch.nn.MSELoss(),
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )

            dict_config = GaussianDictConfig(
                epochs=400,
                alpha=0.01,
                criterion=torch.nn.MSELoss(),
                optimizer_factory=lambda params, lr: torch.optim.AdamW(params, lr=lr), # USAR adamW
                mu_epochs=10, 
                rho_epochs=10, 
                split_mu_rho=False,
                eig_range=[0.05, 0.2],  # ANCHE DE LAS GAUSSIANAS
                regularization_func=GaussianDictLayer.electrostatic_regularization,
                regularization_gamma=0.001,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

            )

            [x1lim, x2lim] = function_limits[function.__name__]
            partition_config = UniformPartitionConfig(
                T=1,
                initial_bounds=torch.tensor([[x1lim for i in range(dim)], [x2lim for i in range(dim)]], dtype=torch.float32),
                activity_threshold=0, overlap_ratio=0.1
            )

            ssesm_config = SSESMConfig(
                n_features= dim, 
                model_epochs=500,
                sparse_coding_config=sparse_coding_config,
                dict_config=dict_config,
                partition_config=partition_config,
                log_interval=100,
                permutation_times=1,
                seed=45,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )

            # 1) Diccionarios principales
            all_metrics = defaultdict(list)       



            # Se recalcula el tamaño del dataset en cada dimension
            n_samples_dim = [int(n**dim) for n in n_samples]
            for n in n_samples_dim:

                # Diccionarios temporales para este chunk
                chunk_metrics = defaultdict(list) # {}

                for j in range(num_runs_per_set):
                    
                    logging.info(rf"""
                    =================================================================================
                                              ,___,      Repetition:      {j+1}/{num_runs_per_set}
                          0/     (\(\         (O.o)      Dataset size:    {n}                      
                         <|      (-.-)        /),,)      Dimension:       {dim}             
                         / \     o_(")(")      " "       Function:        {function.__name__}    
                    =================================================================================
                    """)
                    torch.manual_seed(j)
                    # Generar dataset con n muestras, de dim dimensiones con la funcion function
                    dataset_config = {
                        "n_samples": n,
                        "n_dimensions":dim, 
                        "function": function,
                        "limits": function_limits[function.__name__]}
                    train_data, _, _, test_data, _, _ = generate_custom_nd_function_dataset(**dataset_config )



                    # Crear experimento y entrenar con el metodo train_all y testear con test_all
                    # Se entrenan y testean 4 modelos j veces 
                    experiment = EXPERIMENT(svr_config, nn_config, ssesm_config, pf_config)
                    experiment.setup_dataset(train_data, test_data)
                    experiment.train_all()
                    experiment.test_all()
                    metrics = experiment.metrics


                    # Guardar mse y mae de cada modelo j veces
                    for key, value in metrics.items(): 
                        chunk_metrics[key].append(value)

                # Guardar resultados finales del chunk
                for key, value in chunk_metrics.items():
                    all_metrics[key].append(value)


            all_metrics_dim[dim]=all_metrics
        

        joblib.dump(all_metrics_dim, "./plots/metrics/metrics_"+str(function.__name__)+".joblib")
        joblib.dump(n_samples, "./plots/metrics/n_samples.joblib")
        
        
    # GUARDAR CONFIGURACIONES DEL EXPERIMENTO
    experiment_config_data = {
        "Métricas": ["mae", "mse", "time"],
        "Dimensiones": dimensions,  # Se toma de la variable en el script
        "Repeticiones": num_runs_per_set, # Se toma de la variable en el script
        "Funciones": [func.__name__ for func in functions],
        "Tamaño del dataset 1D": n_samples
    }

    joblib.dump(experiment_config_data, "./plots/config/config_experiment.joblib")
    experiment.save_configs()


    wandb.finish()
    logging.info("Experimento completado. Métricas para boxplots listas.")



if __name__ == "__main__":
    main()
