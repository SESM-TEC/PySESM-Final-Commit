import os
import torch
import numpy as np
import wandb
from pysesm.utils_dataset.generate_dataset import generate_custom_function_dataset
from test_all import test_all
from train_all import train_all


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
    dataset_config = {
        "n_samples": 30,
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
    
    num_runs = 3 # Aumentar el número de corridas para un análisis estadístico más robusto

    # 2. Configuración e Inicio de la Sesión en Weights & Biases
    # ----------------------------------------------------
    # Usa una variable de entorno para mayor seguridad, en lugar de una clave hardcodeada
    wandb_api_key = os.getenv("WANDB_API_KEY") 
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
    else:
        print("Advertencia: No se encontró la variable de entorno WANDB_API_KEY. El registro no funcionará.")

    wandb.init(
        project="PySESM_experiments",
        config={
            "dataset_config": dataset_config,
            "svr_config": svr_config,
            "nn_config": nn_config,
            "num_runs": num_runs
        }
    )

    # 3. Ciclo de Entrenamiento y Recolección de Métricas
    # ----------------------------------------------------
    all_metrics = {
        "NN_MAE": [], "NN_MSE": [],
        "SVR_MAE": [], "SVR_MSE": []
    }

    for i in range(num_runs):
        print(f"--- Corriendo experimento {i + 1}/{num_runs} ---")
        
        # Generar un nuevo dataset en cada corrida para validar la robustez
        train_data, _, _, test_data, _, _ = generate_custom_function_dataset(**dataset_config)
        
        # Entrenar y evaluar los modelos
        train_all(train_data, test_data, svr_config, nn_config)
        
        # El flag de plot solo se activa en la última iteración
        plot_flag = (i == num_runs - 1)
        metrics = test_all(train_data, test_data, plot_flag=plot_flag)
        
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