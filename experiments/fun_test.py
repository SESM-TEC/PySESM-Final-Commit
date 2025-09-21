import fun
from pysesm.utils_dataset.generate_dataset import generate_custom_nd_function_dataset
import matplotlib.pyplot as plt


function_limits = {
    "zakharov_function" : [-10, 10],
    "zhou_function": [0, 1],
    "styblinski_tang_function": [-5, 5],
}
functions=[
    fun.zakharov_function,
    fun.zhou_function, 
    fun.styblinski_tang_function
]


# Crear un subplot de 1 fila y 3 columnas
# La 'projection' 3d es necesaria porque tus datos de entrada son de 2 dimensiones
fig, axes = plt.subplots(
    nrows=1, ncols=len(functions), figsize=(15, 6), 
    subplot_kw={'projection': '3d'}, dpi=100
)

# Iterar sobre las funciones con un índice
for i, function in enumerate(functions):
    # Generar el dataset
    dataset_config = {
        "n_samples": 100,
        "n_dimensions": 2, 
        "function": function,
        "limits": function_limits[function.__name__]
    }
    train_data, _, _, test_data, _, _ = generate_custom_nd_function_dataset(**dataset_config)

    # Convertir los tensores a numpy para graficar
    xtest = test_data["X"].numpy()  # numpy array de shape (2500, 2)
    ztest = test_data["Z"].numpy()  # numpy array de shape (2500,)
    
    # Dibujar en el subplot actual (axes[i])
    ax = axes[i]
    
    ax.scatter(xtest[:, 0], xtest[:, 1], ztest,
               s=10, c=ztest, cmap= 'viridis', marker=".", label="Datos")
    
    # Configurar títulos y etiquetas para cada subplot
    ax.set_title(function.__name__.replace("_", " ").title())
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("Z")
    
plt.tight_layout() # Ajusta automáticamente los subplots para que no se superpongan
plt.show()