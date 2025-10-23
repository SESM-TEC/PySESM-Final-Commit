import matplotlib.pyplot as plt
import joblib   
import os



def plot_caja_bigote(metricas: dict, n_samples: list, filename: str, dim = 2):
    global function
    """
    Crea un conjunto de boxplots para cada métrica en un diccionario.
    Cada subplot representa una métrica (ej. MSE_NN) y contiene múltiples
    cajas, donde cada caja corresponde a un vector de resultados de entrenamientos.
    """
    ancho = 4
    alto = (len(metricas) + ancho - 1) // ancho  # ceil(len/4)
    fig, axes = plt.subplots(nrows=alto, ncols=ancho, figsize=(ancho*4, alto*4), dpi=300)
    fig.suptitle(str(dim) + "D", fontsize=16)  # Título general

    axes = axes.flatten()

    metricas_lista = list(metricas.items())

    for i, (nombre_metrica, datos_metrica) in enumerate(metricas_lista):

        # Calcular ylim para cada bloque de 4 métricas
        start_block = (i // 4) * 4
        end_block = min(start_block + 4, len(metricas_lista))
        block_metrics = metricas_lista[start_block:end_block]

        # calcular máximo de medias de este bloque
        block_max_mean = max(
            sum(v)/len(v)
            for _, val in block_metrics
            for v in val
        )

        # Boxplot
        box = axes[i].boxplot(datos_metrica, patch_artist=True)
        for patch in box['boxes']:
            patch.set_facecolor('lightgreen')
        for median in box['medians']:
            median.set(color='red', linewidth=2)

        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].set_ylim([0, block_max_mean * 1.1])  # 10% extra para margen

        n_samples_dim = [int(n**dim) for n in n_samples]
        axes[i].set_xticklabels(n_samples_dim)

        axes[i].set_title(nombre_metrica)
        axes[i].set_ylabel(nombre_metrica)
        axes[i].set_xlabel('Training samples')
        axes[i].yaxis.grid(True, alpha=0.7)

    # Limpiar ejes extra si hay
    for j in range(len(metricas_lista), len(axes)):
        fig.delaxes(axes[j])

    output_dir = os.path.join(os.getcwd(), rf"./plots/{function}")
    os.makedirs(output_dir, exist_ok=True)
    full_filename = os.path.join(output_dir, f"{filename}_{dim}D.png")

    plt.tight_layout()
    plt.savefig(full_filename, dpi=300)

    #wandb.log({"Boxplots": wandb.Image(fig)})


# modelos = [svr, nn, ssesm, pf]s
# metricas = [mae_modelo, mse_modelo, time_modelo]
functions=['function_zhou', 'function_zakharov', 'function_styblinski_tang']
for function in functions:
    
    metrics = joblib.load(f"./metrics/metrics_{function}.joblib")
    n_samples = joblib.load("./metrics/n_samples.joblib")

    for dim, dim_metrics in metrics.items():
        plot_caja_bigote(dim_metrics, n_samples, "metrics", dim=dim)
