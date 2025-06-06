import pytest
import torch
import numpy as np
import logging

from pysesm.models.SSESM import SSESM, SSESMConfig
from pysesm.blocks.UniformPartitionManager import UniformPartitionConfig
from pysesm.dictionaries import GaussianDictConfig # O cualquier DictConfig
from pysesm.sparse_coding import ISTAConfig # O cualquier SparseCodingConfig
from pysesm.device_manager.DeviceManager import DeviceManager
from pysesm.enums.DeviceTargetEnum import DeviceTarget

# --- Logger y Fixtures (puedes adaptarlos o usar los que ya tienes) ---
logger = logging.getLogger("test_amplitude_prediction")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

@pytest.fixture(scope="module")
def device_manager_fixture():
    # Ajusta los dispositivos según sea necesario
    device_map = {
        DeviceTarget.GLOBAL: "cpu",
        DeviceTarget.SPARSE_CODING_LAYER: "cpu",
        DeviceTarget.DICTIONARY_LAYER: "cpu",
        DeviceTarget.PARTITION_MANAGER: "cpu"
    }
    return DeviceManager(logger=logger, default_device="cpu", device_map=device_map)

# --- Clase MockableSSESM para la Prueba ---
class MockableSSESM(SSESM):
    def __init__(self, config: SSESMConfig, logger: logging.Logger, device_manager: DeviceManager, **kwargs):
        super().__init__(config, logger, device_manager, **kwargs)
        # Este diccionario mapeará block_index a la salida mockeada para ese bloque
        self.mock_eval_outputs_per_block = {}

    def set_mock_eval_output(self, block_index_tuple, output_value):
        """Define la salida mockeada para un bloque específico."""
        # Aseguramos que la salida sea un tensor 2D (n_samples_in_block, 1)
        self.mock_eval_outputs_per_block[block_index_tuple] = torch.tensor([[output_value]],
                                                                       device=self.device_manager.get_device(DeviceTarget.DICTIONARY_LAYER),
                                                                       dtype=torch.float32)

    # Sobrescribimos _predict_block para interceptar y usar el mock si está definido
    # Esto es más directo que mockear evaluation_func, ya que _predict_block es llamado por SSESM.predict
    def _predict_block(self, block, dictionary_shape=None, custom_h=None) -> torch.Tensor:
        block_idx_tuple = block.block_index # Asumiendo que block_index es una tupla
        if block_idx_tuple in self.mock_eval_outputs_per_block:
            # Devolvemos la salida mockeada para este bloque.
            # El valor mockeado ya está en la "escala de target".
            # Aseguramos que el número de "muestras" en la salida mock coincida con X.
            # Si block.normalized_X tiene N muestras, la salida debe ser (N,1)
            num_samples_in_block = block.normalized_X.shape[0]
            mock_output_for_one_sample = self.mock_eval_outputs_per_block[block_idx_tuple]
            
            # Si el mock es para una sola muestra, lo repetimos si hay más.
            # Esto es una simplificación; en un caso real, el mock debería ser más sofisticado
            # o la prueba debería usar un solo punto de prueba por bloque.
            if num_samples_in_block > 1 and mock_output_for_one_sample.shape[0] == 1:
                 return mock_output_for_one_sample.repeat(num_samples_in_block, 1)
            return mock_output_for_one_sample
        else:
            # Si no hay mock para este bloque, usar la implementación original.
            return super()._predict_block(block, dictionary_shape, custom_h)


def test_ssesm_predict_amplitude_denormalization(device_manager_fixture):
    """
    Verifica que SSESM.predict use correctamente el factor de amplitud
    para desnormalizar la predicción cruda del modelo.
    Usa el escenario de espacio de entrada de -2 a 2 y 2x2 bloques.
    """
    device = device_manager_fixture.get_device(DeviceTarget.GLOBAL)

    # 1. Configuración del SSESM (con MockableSSESM)
    n_features = 2
    n_functions = 3 # Necesario para las configs, valor no crítico para esta prueba

    # Configuración de partición: -2 a 2, 2x2 bloques
    # Esto significa que block_size será [2,2]
    partition_conf = UniformPartitionConfig(
        T=torch.tensor([2, 2], device=device, dtype=torch.int),
        initial_bounds=np.array([[-2.0, -2.0], [2.0, 2.0]], dtype=np.float32),
        activity_threshold=0
    )
    dict_conf = GaussianDictConfig(epochs=1, alpha=0.1, eig_range=[0.1,1.0], mu_range=[-1.0,1.0])
    sc_conf = ISTAConfig(n_functions=n_functions, epochs=1, alpha=0.1, lambd=0.01)

    ssesm_conf = SSESMConfig(
        n_features=n_features,
        model_epochs=1, # No entrenaremos realmente el diccionario/h
        sparse_coding_config=sc_conf,
        dict_config=dict_conf,
        partition_config=partition_conf,
        permutation_times=1
    )
    model = MockableSSESM(config=ssesm_conf, logger=logger, device_manager=device_manager_fixture)

    # 2. Datos de "Entrenamiento" para establecer amplitudes
    # Bloque (0,0): Origen conceptual [-2,-2]. Datos y>1 -> amplitude < 1
    # Bloque (1,1): Origen conceptual [0,0]. Datos y<=1 -> amplitude = 1
    X_train = torch.tensor([
        [-1.0, -1.0], # Para bloque (0,0)
        [ 1.0,  1.0]  # Para bloque (1,1)
    ], device=device, dtype=torch.float32)
    
    y_train_block00 = torch.tensor([[5.0]], device=device, dtype=torch.float32) # max_abs=5 -> amplitude=0.2
    y_train_block11 = torch.tensor([[0.8]], device=device, dtype=torch.float32) # max_abs=0.8 -> amplitude=1.0
    y_train = torch.cat((y_train_block00, y_train_block11), dim=0)

    # 3. "Entrenar" (solo para procesar puntos y calcular amplitudes)
    # Esto llama a manager.add_points() que calcula block.amplitude y block.target
    model.partition_manager.add_points(X_train, y_train)
    
    # Verificar amplitudes en los bloques de "entrenamiento"
    block00_train = model.partition_manager.blocks[0,0]
    block11_train = model.partition_manager.blocks[1,1]

    assert block00_train.is_active
    assert block11_train.is_active
    assert pytest.approx(block00_train.amplitude) == 1.0 / 5.0 # 0.2
    assert pytest.approx(block00_train.target[0].item()) == 1.0 # 5.0 * 0.2
    
    assert pytest.approx(block11_train.amplitude) == 1.0
    assert pytest.approx(block11_train.target[0].item()) == 0.8 # 0.8 * 1.0

    # Necesario para que retrieve_test_active_blocks copie las capas SC (aunque no las usemos activamente)
    model.partition_manager.init_sparse_coding_per_block(config=ssesm_conf.sparse_coding_config, evaluation_func=model.evaluation_func)


    # 4. Datos de Prueba (un punto para cada bloque que "entrenamos")
    X_test = torch.tensor([
        [-1.5, -0.5], # Mapea a bloque (0,0)
        [ 0.5,  1.5]  # Mapea a bloque (1,1)
    ], device=device, dtype=torch.float32)
    # y_test es necesario para retrieve_test_active_blocks, pero su valor no afecta la predicción final
    y_test = torch.tensor([[5.0], [0.8]], device=device, dtype=torch.float32)

    # 5. Configurar las Salidas Mockeadas del Modelo
    # Asumimos que el modelo, antes de la desnormalización por amplitud, predice perfectamente los "targets"
    # Para el punto en el bloque (0,0), el target aprendido fue 1.0
    # Para el punto en el bloque (1,1), el target aprendido fue 0.8
    model.set_mock_eval_output(block_index_tuple=(0,0), output_value=1.0) # Predicción cruda para bloque (0,0)
    model.set_mock_eval_output(block_index_tuple=(1,1), output_value=0.8) # Predicción cruda para bloque (1,1)

    # 6. Llamar a predict()
    y_final_predictions = model.predict(X_test, y_test)

    # 7. Verificar las predicciones finales
    # Para el primer punto de prueba (mapeado al bloque (0,0)):
    #   - Amplitud del bloque (0,0) es 0.2.
    #   - Predicción cruda mockeada para este bloque es 1.0.
    #   - Predicción final = PredicciónCruda / Amplitud = 1.0 / 0.2 = 5.0.
    # Para el segundo punto de prueba (mapeado al bloque (1,1)):
    #   - Amplitud del bloque (1,1) es 1.0.
    #   - Predicción cruda mockeada para este bloque es 0.8.
    #   - Predicción final = PredicciónCruda / Amplitud = 0.8 / 1.0 = 0.8.

    expected_final_predictions = torch.tensor([5.0,0.8], device=device, dtype=torch.float32)

    assert y_final_predictions.shape == expected_final_predictions.shape
    assert torch.allclose(y_final_predictions, expected_final_predictions, atol=1e-5), \
        f"Predicciones finales incorrectas. Esperado: {expected_final_predictions}, Obtenido: {y_final_predictions}"
