import numpy as np
import torch
from pysesm.models.ISTALayer import ISTALayer
from collections import defaultdict


class SubBlock:
    """
    Represents a sub-block in a 2D grid.

    Attributes:
    - vertices (np.ndarray): The vertices of the sub-block.
    - amplitude (int): The amplitude of the sub-block.
    - samples_inside (list): List of samples inside the sub-block.
    - output_values (list): List of output values.
    - index (list): List of index of the original point X

    Methods:
    - add_point(point): Add a point to the sub-block.
    """

    def __init__(self, amplitude=1, ista_layer=None):
        self.amplitude = amplitude
        self.ista_layer = ista_layer
        self._X = []
        self._index = []
        self.predicted_output = []
        self.output_values = []

    def add_point(self, point, y):
        self._X.append(point)
        self.output_values.append(y)

    def get_X(self):
        return self._X

    def set_X(self, new_X):
        self._X = new_X


def get_sub_block_vertices(grid_size, row, col):
    """
    Get the vertices of a sub-block in a 2D grid.

    Args:
    - grid_size (int): The number of segments per dimension.
    - row (int): The row index of the sub-block.
    - col (int): The column index of the sub-block.

    Returns:
    np.ndarray: The vertices of the sub-block.
    """
    delta = 1.0 / grid_size
    x0 = col * delta
    x1 = (col + 1) * delta
    y0 = row * delta
    y1 = (row + 1) * delta
    return np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]])


def locate_samples_in_sub_blocks(x_n, y, t, T):
    """
    Locate points in their respective sub-blocks in a 2D grid.

    Args:
    - x_n (np.ndarray): The normalized points between 0 and 1.
    - y (np.narray) : The output values associated with the samples
    - t (np.ndarray): The integer part of the normalized points.
    - T (int): The number of segments per dimension.

    Returns:
    np.ndarray: Array of SubBlock instances representing the sub-blocks.
    """

    sub_blocks = np.empty((T * T), dtype=object)

    for index in range(T * T):
        sub_blocks[index] = SubBlock()

    for i in range(x_n.shape[0]):
        point = x_n[i]
        location = t[i]
        sub_block = sub_blocks[location[0] * T + location[1]]
        sub_block.add_point(point, y[i])

    return sub_blocks


def generate_list_of_subblock(
    sub_blocks, l_functions, weight_decay, alpha, lambd
):
    """
    Generate a list for all the sub-blocks with their expected squeeze factor.

    Arg:
      np.ndarray: Array of SubBlock instances representing the sub-blocks.
      float: Weight decay for the optimizer in ISTALayer.
      float: Learning rate alpha.
      float: Regularization parameter lambda.

    Returns:
    list: List of all sub-blocks with their block.output_values modified.
    """
    list_sub_blocks = []
    for block in sub_blocks:
        print(f"OUTPUT VALUES: {len(block.output_values)}")
        if len(block.output_values) != 0:
            block.amplitude = squeze_factor(block.output_values)
            block.ista_layer = ISTALayer(
                n_functions=l_functions,
                weight_decay=weight_decay,
                alpha=alpha,
                lambd=lambd,
            )
            block.output_values = [
                value * block.amplitude for value in block.output_values
            ]

            list_sub_blocks.append(block)

    return list_sub_blocks

def predict_on_test_set(X_test, model, T, train_sb):
    """
    Predicts on a test set using a given model and training sub-blocks.

    Args:
    - X_test (torch.Tensor): A tensor containing the test data.
    - model (Model): The model used for prediction.
    - T (int): The scaling factor for normalizing the data.
    - train_sb (list): A list of training sub-blocks used for prediction.

    Returns:
    - sorted_predictions (torch.Tensor): A tensor containing the sorted predictions for the test data.
    """
    t_test, x_n_test = data_mapping(X_test, T)

    sorted_predictions = torch.zeros(
        len(X_test), dtype=torch.float32
    )  # Tensor para almacenar las predicciones ordenadas

    for row in range(T):
        for col in range(T):
            sub_block_points = x_n_test[(t_test[:, 0] == row) & (t_test[:, 1] == col)]
            indices = np.where((t_test[:, 0] == row) & (t_test[:, 1] == col))[0]

            if len(sub_block_points) > 0:
                X_sub_block = torch.tensor(sub_block_points, dtype=torch.float32)

                try:
                    current_train_block = train_sb[row * T + col]
                    predictions_sub_block = model.predict(
                        X_sub_block, current_train_block.ista_layer
                    )
                    print(
                        "CURRENT ISTA on sub block ", current_train_block.ista_layer.h
                    )
                    print("SUM VALUES H ", current_train_block.ista_layer.h.data.sum())
                    sorted_predictions[indices] = (
                        predictions_sub_block.float() / current_train_block.amplitude
                    )
                except IndexError:
                    # No se hace nada para los bloques dummy
                    pass

    return sorted_predictions


def predict_on_test_set_bsesm(X_test, model, T, train_sb):
    """
    Predicts on a test set using a given model and training sub-blocks.

    Args:
    - X_test (torch.Tensor): A tensor containing the test data.
    - model (Model): The model used for prediction.
    - T (int): The scaling factor for normalizing the data.
    - train_sb (list): A list of training sub-blocks used for prediction.

    Returns:
    - sorted_predictions (torch.Tensor): A tensor containing the sorted predictions for the test data.
    """
    t_test, x_n_test = data_mapping(X_test, T)

    sorted_predictions = torch.zeros(
        len(X_test), dtype=torch.float32
    )  # Tensor para almacenar las predicciones ordenadas

    for row in range(T):
        for col in range(T):
            sub_block_points = x_n_test[(t_test[:, 0] == row) & (t_test[:, 1] == col)]
            indices = np.where((t_test[:, 0] == row) & (t_test[:, 1] == col))[0]

            if len(sub_block_points) > 0:
                X_sub_block = torch.tensor(sub_block_points, dtype=torch.float32)

                try:
                    current_train_block = train_sb[row * T + col]
                    predictions_sub_block = model.predict(
                        X_sub_block, current_train_block.ista_layer
                    )
                    print(
                        "CURRENT ISTA on sub block ", current_train_block.ista_layer.h
                    )
                    print("SUM VALUES H ", current_train_block.ista_layer.h.data.sum())
                    sorted_predictions[indices] = (
                        predictions_sub_block.float() / current_train_block.amplitude
                    )
                except IndexError:
                    # No se hace nada para los bloques dummy
                    pass

    return sorted_predictions


def count_unique_combinations(T):
    """
    Count the unique combinations in the list generated by the data_mapping function.

    Args:
        T (int): Value T.

    Returns:
        dict: Dictionary with unique combinations as keys and the number of occurrences as values.
    """
    # Crear un diccionario para contar las combinaciones únicas
    conteo_combinaciones = defaultdict(int)

    # Contar las combinaciones únicas en T
    for combinacion in T:
        combinacion_tuple = tuple(combinacion)
        conteo_combinaciones[combinacion_tuple] += 1

    # Imprimir el conteo de combinaciones únicas
    for combinacion, cantidad in conteo_combinaciones.items():
        print(f"Combinación {combinacion}: {cantidad} veces")


def squeze_factor(Y):
    """
    Calculates a squeezing factor for a given set of values.

    Args:
    - Y (iterable): An iterable containing numeric values.

    Returns:
    - float: The squeezing factor. If the maximum value in Y is greater than 1, returns the reciprocal of the maximum value. Otherwise, returns 1.0.
    """
    e_f = 0.0
    max_y = max(Y)
    if max_y > 1:
        e_f = 1 / max_y
    else:
        e_f = 1.0
    return e_f
