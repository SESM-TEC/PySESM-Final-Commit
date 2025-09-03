import matplotlib.pyplot as plt
import numpy as np

def plot_caja_bigote(metricas: dict):
    """
    Crea un conjunto de boxplots para cada métrica en un diccionario.
    Cada subplot representa una métrica (ej. MSE_NN) y contiene múltiples
    cajas, donde cada caja corresponde a un vector de resultados de entrenamientos.

    Args:
        metricas (dict): Diccionario donde las claves son los nombres de las métricas
                         y los valores son listas de vectores.
                         Ej: {'MSE_NN': [vector_chunk1, vector_chunk2, ...]}
    """
    ancho = len(metricas) // 2
    alto = 2
    
    # 1. Quitar la proyección 3D, ya que es un plot 2D
    fig, axes = plt.subplots(nrows=alto, ncols=ancho, figsize=(8, 8))
    
    # El método axes.flatten() es útil para trabajar con una matriz de ejes
    axes = axes.flatten()

    # 2. Iterar sobre el diccionario usando enumerate para obtener un índice
    for i, (nombre_metrica, datos_metrica) in enumerate(metricas.items()):
        
        # 3. Crear el boxplot para los datos de la métrica actual
        # `datos_metrica` es una lista de vectores, perfecta para boxplot
        axes[i].boxplot(datos_metrica)
        
        # Opcional: Establecer etiquetas para las cajas.
        # Si las cajas representan "chunks", puedes nombrarlas así.
        n_chunks = len(datos_metrica)
        labels = [f'Chunk {j+1}' for j in range(n_chunks)]
        axes[i].set_xticklabels(labels)
        
        # 4. Configurar el título y las etiquetas de los ejes
        axes[i].set_title(nombre_metrica)
        axes[i].set_ylabel(nombre_metrica)
        axes[i].set_xlabel('Training samples')
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.show()

# Ejemplo de uso con un diccionario de datos simulado
# Cada lista interna representa un "chunk" de datos de entrenamiento
metricas_ejemplo = {
    'MSE_NN': [np.random.rand(10), np.random.rand(10)*0.8, np.random.rand(10)*0.6],
    'MSE_SVR': [np.random.rand(10)*1.2, np.random.rand(10), np.random.rand(10)*0.7],
    'MAE_NN': [np.random.rand(10)*0.5, np.random.rand(10)*0.4, np.random.rand(10)*0.3],
    'MAE_SVR': [np.random.rand(10)*0.6, np.random.rand(10)*0.5, np.random.rand(10)*0.4]
}

plot_caja_bigote(metricas_ejemplo)