import matplotlib.pyplot as plt
import joblib   

def plot_caja_bigote(metricas: dict, n_samples: list, filename: str, ylim = None):
    """
    Crea un conjunto de boxplots para cada métrica en un diccionario.
    Cada subplot representa una métrica (ej. MSE_NN) y contiene múltiples
    cajas, donde cada caja corresponde a un vector de resultados de entrenamientos.

    Args:
        metricas (dict): Diccionario donde las claves son los nombres de las métricas
                        y los valores son listas de vectores.
                        Ej: {'MSE_NN': [vector_chunk1, vector_chunk2, ...]}
        n_samples (list): Lista con el número de muestras usadas. Ej: [8, 16, 32, ...]
    """
    ancho = len(metricas) // 2
    alto = 2
    fig, axes = plt.subplots(nrows=alto, ncols=ancho, figsize=(ancho*4, alto*4), dpi=300)
    
    # El método axes.flatten() es útil para trabajar con una matriz de ejes
    axes = axes.flatten()
    # 2. Iterar sobre el diccionario usando enumerate para obtener un índice
    for i, (nombre_metrica, datos_metrica) in enumerate(metricas.items()):
            
        # 3. Crear el boxplot para los datos de la métrica actual
        # `datos_metrica` es una lista de vectores, perfecta para boxplot
        box = axes[i].boxplot(datos_metrica, patch_artist = True)
        
        for patch in box['boxes']:
            patch.set_facecolor('lightgreen')
        for median in box['medians']:
            median.set(color='red', linewidth=2)
        
        # 4. Configurar el título y las etiquetas de los ejes
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].set_ylim(ylim)
        axes[i].set_xticklabels(n_samples)
        axes[i].set_title(nombre_metrica)
        axes[i].set_ylabel(nombre_metrica)
        axes[i].set_xlabel('Training samples')
        axes[i].yaxis.grid(True, alpha=0.7)

    plt.tight_layout()
    plt.savefig(filename+".png", dpi=300)
    #wandb.log({"Boxplots": wandb.Image(fig)})

times   = joblib.load("all_times.joblib")
metrics = joblib.load("all_metrics.joblib")
n_samples = joblib.load("n_samples.joblib")

plot_caja_bigote(metrics, n_samples, "all_metrics", ylim=(0, 1))
plot_caja_bigote(times, n_samples, "all_times")
