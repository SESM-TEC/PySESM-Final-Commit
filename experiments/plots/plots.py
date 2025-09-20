import matplotlib.pyplot as plt
import joblib   
import os

def plot_caja_bigote(metricas: dict, n_samples: list, filename: str, ylim = None, dim = 2):
    global function
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
    fig.suptitle(str(dim) + "D", fontsize=16)  # Título general

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
        axes[i].set_ylim([0, ylim])

        n_samples_dim = [int(n**dim) for n in n_samples]
        axes[i].set_xticklabels(n_samples_dim)

        axes[i].set_title(nombre_metrica)
        axes[i].set_ylabel(nombre_metrica)
        axes[i].set_xlabel('Training samples')
        axes[i].yaxis.grid(True, alpha=0.7)

    output_dir = os.path.join(os.getcwd(), function) 
    os.makedirs(output_dir, exist_ok=True)
    name = filename+"_"+str(dim)+ "D"+".png"
    full_filename = os.path.join(output_dir, name)

    plt.tight_layout()
    plt.savefig(full_filename, dpi=300)
    #wandb.log({"Boxplots": wandb.Image(fig)})

def calc_max_mean(metricas: dict):
    mean_max = 0
    for _, value in metricas.items():
        for v in value:
            mean_v = sum(v)/len(v)
            if mean_v > mean_max:
                mean_max = mean_v
    return mean_max


functions=['zakharov_function', 'rosenbrock_rescaled_function', 'zhou_function']
for function in functions:
    
    times   = joblib.load(f"all_times{function}.joblib")
    metrics = joblib.load(f"all_metrics{function}.joblib")
    n_samples = joblib.load("n_samples.joblib")

    for dim, dim_metrics in metrics.items():
        max_mean = calc_max_mean(dim_metrics)
        plot_caja_bigote(dim_metrics, n_samples, "metrics", ylim=max_mean, dim=dim)

    for dim, dim_times in times.items():
        max_mean = calc_max_mean(dim_times)
        plot_caja_bigote(dim_times, n_samples, "times", ylim= max_mean, dim=dim)