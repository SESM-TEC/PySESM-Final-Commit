#  Reporte de Experimento PySESM

## Configuración de los modelos
| Modelo | Configuración |
| :--- | :--- |
| **SVR (Support Vector Regressor)** | svr_config |
| **PF (Polynomial Features)** | pf_config |
| **NN (Neural Network)** | nn_config |
| **SESM (Sparse Encoding Surrogate Model)** | sesm_config |

## Configuración del experimento
| Categoría | Elementos |
| :--- | :--- |
| **Métricas** | mae, mse, time |
| **Dimensiones** | [1, 2] |
| **Repeticiones** | 10 |
| **Funciones** | zakharov, styblinski tang, zhou |
| **Tamaño del dataset 1D** | [4, 8, 16, 32, 64] |











---

## function_zhou

<img src="plots/function_zhou/metrics_1D.png" alt="Metricas en 1 dimension" width="650">  
<img src="plots/function_zhou/metrics_2D.png" alt="Metricas en 2 dimensiones" width="650">  

---

## function_zakharov

<img src="plots/function_zakharov/metrics_1D.png" alt="Metricas en 1 dimension" width="650">  
<img src="plots/function_zakharov/metrics_2D.png" alt="Metricas en 2 dimensiones" width="650">  
---

## function_styblinski_tang
<img src="plots/function_styblinski_tang/metrics_1D.png" alt="Metricas en 1 dimension" width="650">  
<img src="plots/function_styblinski_tang/metrics_2D.png" alt="Metricas en 2 dimensiones" width="650">  

