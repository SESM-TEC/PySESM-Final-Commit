"""
Normalization Logic Tests.

Specific tests for coordinate normalization within blocks, covering edge cases
like boundaries, negative origins, and conceptual limits.

Copyright (c) 2023-2025, Tecnológico de Costa Rica
All rights reserved.

This source code is licensed under the BSD 3-Clause License found in the
LICENSE file in the root directory of this source tree.

SPDX-License-Identifier: BSD-3-Clause
"""
import pytest
import torch
import numpy as np
import logging

from pysesm.blocks.PartitionBlock import PartitionBlock
from pysesm.blocks.UniformPartitionManager import UniformPartitionManager, UniformPartitionConfig
from pysesm.sparse_coding.ISTALayer import ISTAConfig # Para inicializar SC layers

# Configuración básica de logger para las pruebas
logger = logging.getLogger("test_normalization_pysesm_v2")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# --- Fixtures Comunes ---
@pytest.fixture(scope="module")
def device_fixture():
    """Proporciona el device para las pruebas."""
    return "cpu"
    
@pytest.fixture
def partition_block_factory(device_fixture):
    """Factory para crear instancias de PartitionBlock."""
    def _create_block(space_origin_coords, block_idx_tuple, block_size_coords, custom_eps=None): # custom_eps for specific tests
        device = device_fixture
        space_origin = torch.tensor(space_origin_coords, device=device, dtype=torch.float32)
        block_size = torch.tensor(block_size_coords, device=device, dtype=torch.float32)

        # Calculate block_scope explicitly for the test factory
        idx_tensor = torch.tensor(block_idx_tuple, device=device, dtype=torch.float32)
        # Calculate position relative to space_origin
        base_pos = space_origin + idx_tensor * block_size
        block_scope = torch.stack((base_pos, base_pos + block_size))
        
        return PartitionBlock(block_index=block_idx_tuple, block_size=block_size, block_scope=block_scope, device=device, space_origin=space_origin)

    return _create_block

@pytest.fixture
def uniform_manager_factory(device_fixture):
    """Factory para crear instancias de UniformPartitionManager."""
    def _create_manager(T_val, initial_bounds_val_np, threshold_val=0):
        config = UniformPartitionConfig(
            T=T_val,
            initial_bounds=initial_bounds_val_np,
            activity_threshold=threshold_val,
            device=device_fixture
        )
        return UniformPartitionManager(
            config=config,
            logger=logger
        )
    return _create_manager

# --- Pruebas para PartitionBlock.normalize_points() ---

def test_pb_normalize_basic_conceptual(partition_block_factory):
    """Prueba la normalización básica en un PartitionBlock (límites conceptuales)."""
    block = partition_block_factory([0.0, 0.0], (0,0), [1.0, 1.0])
    point_x_orig = torch.tensor([0.5, 0.25], device=block.device)
    block.new_point(point_x_orig, torch.tensor([1.0], device=block.device), 0)
    block.normalize_points() # Asumiendo que usa el origen conceptual

    # Origen conceptual: [0,0]. Tamaño: [1,1]
    # normalized = (point_x_orig - conceptual_origin) / block_size
    #            = ([0.5, 0.25] - [0,0]) / [1.0, 1.0]
    #            = [0.5, 0.25]
    expected_normalized = torch.tensor([0.5, 0.25], device=block.device)
    assert block.normalized_X is not None
    assert torch.allclose(block.normalized_X.get_for_device(block.device)[0], expected_normalized, atol=1e-7)

def test_pb_normalize_negative_origin_conceptual(partition_block_factory):
    """Prueba la normalización con origen de espacio negativo (límites conceptuales)."""
    block = partition_block_factory([-2.0, -2.0], (0,0), [1.0, 1.0])
    point_x_orig = torch.tensor([-1.5, -1.75], device=block.device) # Dentro del bloque
    block.new_point(point_x_orig, torch.tensor([1.0], device=block.device), 0)
    block.normalize_points()

    # Origen conceptual del bloque (0,0): [-2,-2]. Tamaño: [1,1]
    # normalized = ([-1.5, -1.75] - [-2,-2]) / [1.0, 1.0]
    #            = [0.5, 0.25]
    expected_normalized = torch.tensor([0.5, 0.25], device=block.device)
    assert block.normalized_X is not None
    assert torch.allclose(block.normalized_X.get_for_device(block.device)[0], expected_normalized, atol=1e-7)

def test_pb_normalize_mixed_origin_and_index_conceptual(partition_block_factory):
    """Prueba la normalización con origen de espacio negativo e índice > 0 (límites conceptuales)."""
    block = partition_block_factory([-10.0, -10.0], (1,1), [2.0, 2.0])
    point_x_orig = torch.tensor([-7.0, -7.5], device=block.device) # Dentro del bloque (1,1)
    block.new_point(point_x_orig, torch.tensor([1.0], device=block.device), 0)
    block.normalize_points()

    # Origen conceptual del bloque (1,1): [-10,-10] + (1,1)*[2,2] = [-8,-8]. Tamaño: [2,2]
    # normalized = ([-7.0, -7.5] - [-8,-8]) / [2.0, 2.0]
    #            = ([1.0, 0.5]) / [2.0, 2.0]
    #            = [0.5, 0.25]
    expected_normalized = torch.tensor([0.5, 0.25], device=block.device)
    assert block.normalized_X is not None
    assert torch.allclose(block.normalized_X.get_for_device(block.device)[0], expected_normalized, atol=1e-7)

def test_pb_normalize_points_at_conceptual_boundaries_strict(partition_block_factory):
    """Prueba la normalización de puntos en los límites conceptuales ESTRICTOS del bloque."""
    block = partition_block_factory([0.0, 0.0], (0,0), [1.0, 1.0]) # Scope conceptual [0,1]x[0,1]

    point_lower = torch.tensor([0.0, 0.0], device=block.device) # Exactamente en origen conceptual
    block.new_point(point_lower, torch.tensor([1.0], device=block.device), 0)

    point_upper = torch.tensor([1.0, 1.0], device=block.device) # Exactamente en fin conceptual
    block.new_point(point_upper, torch.tensor([1.0], device=block.device), 1)
    block.normalize_points()

    expected_lower_norm = torch.tensor([0.0, 0.0], device=block.device)
    assert torch.allclose(block.normalized_X.get_for_device(block.device)[0], expected_lower_norm, atol=1e-7)

    expected_upper_norm = torch.tensor([1.0, 1.0], device=block.device)
    assert torch.allclose(block.normalized_X.get_for_device(block.device)[1], expected_upper_norm, atol=1e-7)

def test_pb_normalize_points_outside_conceptual_inside_scope(partition_block_factory):
    """
    Prueba la normalización de puntos que están FUERA de los límites conceptuales
    pero DENTRO del block_scope (debido a eps).
    Estos deberían normalizarse a < 0 o > 1.
    """
    # Asumimos que PartitionBlock usa torch.finfo(torch.float32).eps internamente
    # Si pudieras pasar un `eps_val` mayor a PartitionBlock, esta prueba sería más robusta.
    # Por ahora, usaremos el eps de la máquina.
    machine_eps = torch.finfo(torch.float32).eps
    block = partition_block_factory([0.0, 0.0], (0,0), [1.0, 1.0]) # Conceptual [0,1]x[0,1]

    # Punto ligeramente ANTES del origen conceptual (pero dentro de block_scope[0] si eps es el de máquina)
    # block_scope[0] es [0-eps, 0-eps]. Origen conceptual es [0,0].
    point_before_conceptual_origin = torch.tensor([0.5 * machine_eps, 0.5 * machine_eps], device=block.device)
    
    # Punto ligeramente DESPUÉS del fin conceptual (pero dentro de block_scope[1])
    # block_scope[1] es [1-eps, 1-eps]. Fin conceptual es [1,1].
    point_after_conceptual_end = torch.tensor([1.0 - 0.5 * machine_eps, 1.0 - 0.5 * machine_eps], device=block.device)

    block.new_point(point_before_conceptual_origin, torch.tensor([1.0], device=block.device), 0)
    block.new_point(point_after_conceptual_end, torch.tensor([1.0], device=block.device), 1)
    block.normalize_points() # Asumiendo que usa el origen conceptual [0,0]

    # Normalización de point_before_conceptual_origin:
    # ([-0.5eps, -0.5eps] - [0,0]) / [1,1] = [-0.5eps, -0.5eps]
    expected_norm_before = torch.tensor([0.5 * machine_eps, 0.5 * machine_eps], device=block.device)
    assert torch.allclose(block.normalized_X.get_for_device(block.device)[0], expected_norm_before, atol=1e-9) # Usar atol más pequeño para eps
    assert torch.all(block.normalized_X.get_for_device(block.device)[0] > 0.0)

    # Normalización de point_after_conceptual_end:
    # ([1-0.5eps, 1-0.5eps] - [0,0]) / [1,1] = [1-0.5eps, 1-0.5eps]
    expected_norm_after = torch.tensor([1.0 - 0.5 * machine_eps, 1.0 - 0.5 * machine_eps], device=block.device)
    assert torch.allclose(block.normalized_X.get_for_device(block.device)[1], expected_norm_after, atol=1e-9)
    assert torch.all(block.normalized_X.get_for_device(block.device)[1] < 1.0)


def test_pb_normalize_zero_block_size_dimension_conceptual(partition_block_factory):
    """Prueba la normalización (conceptual) cuando una dimensión del bloque tiene tamaño cero."""
    block = partition_block_factory([0.0, 0.0], (0,0), [1.0, 0.0])
    point_x_orig = torch.tensor([0.5, 0.0], device=block.device) # y coincide con el origen conceptual en esa dim
    block.new_point(point_x_orig, torch.tensor([1.0], device=block.device), 0)
    
    point_x_orig_y_offset = torch.tensor([0.5, 0.1], device=block.device) # y es diferente del origen conceptual
    block.new_point(point_x_orig_y_offset, torch.tensor([1.0], device=block.device), 1)
    block.normalize_points()

    # Origen conceptual: [0,0]. Tamaño: [1,0]. effective_sizes para normalización: [1,1]
    # Para point_x_orig ([0.5, 0.0]):
    # normalized = ([0.5, 0.0] - [0,0]) / [1,1] (effective_sizes)
    #            = [0.5, 0.0]
    expected_normalized_1 = torch.tensor([0.5, 0.0], device=block.device)
    assert torch.allclose(block.normalized_X.get_for_device(block.device)[0], expected_normalized_1, atol=1e-7)

    # Para point_x_orig_y_offset ([0.5, 0.1]):
    # normalized = ([0.5, 0.1] - [0,0]) / [1,1] (effective_sizes)
    #            = [0.5, 0.1]
    expected_normalized_2 = torch.tensor([0.5, 0.1], device=block.device)
    assert torch.allclose(block.normalized_X.get_for_device(block.device)[1], expected_normalized_2, atol=1e-7)


# --- Pruebas para UniformPartitionManager (enfocadas en retrieve_test_active_blocks) ---

def test_manager_retrieve_test_blocks_normalization_values_conceptual(uniform_manager_factory, device_fixture):
    """
    Prueba exhaustiva de la normalización (conceptual) de datos de prueba en UniformPartitionManager.
    Verifica que X_test se normalice correctamente relativo a los límites conceptuales de cada bloque de prueba.
    """
    device = device_fixture
    
    initial_bounds_np = np.array([[-2.0, -2.0], [2.0, 2.0]], dtype=np.float32)
    T_tensor = torch.tensor([2, 2], device=device, dtype=torch.int)
    manager = uniform_manager_factory(T_val=T_tensor, initial_bounds_val_np=initial_bounds_np)

    X_train = torch.tensor([[-1.0, -1.0], [1.0, 1.0]], device=device, dtype=torch.float32)
    y_train = torch.tensor([[0.5], [0.8]], device=device, dtype=torch.float32)
    manager.add_points(X_train, y_train) # Esto llama a normalize_points internamente para los bloques de entrenamiento

    # Verificación de normalización en bloques de ENTRENAMIENTO (asumiendo que PartitionBlock.normalize_points ya fue corregido)
    train_block_0_0 = manager.blocks[0,0] # Conceptual: x,y en [-2,0]
    # Origen conceptual de bloque (0,0) es [-2,-2]. Punto es [-1,-1]. Tamaño de bloque es [2,2].
    # Normalizado = ([-1,-1] - [-2,-2]) / [2,2] = [1,1]/[2,2] = [0.5,0.5]
    assert torch.allclose(train_block_0_0.normalized_X.get_for_device(device)[0], torch.tensor([0.5,0.5], device=device), atol=1e-7)

    train_block_1_1 = manager.blocks[1,1] # Conceptual: x,y en [0,2]
    # Origen conceptual de bloque (1,1) es [0,0]. Punto es [1,1]. Tamaño de bloque es [2,2].
    # Normalizado = ([1,1] - [0,0]) / [2,2] = [1,1]/[2,2] = [0.5,0.5]
    assert torch.allclose(train_block_1_1.normalized_X.get_for_device(device)[0], torch.tensor([0.5,0.5], device=device), atol=1e-7)

    sc_config = ISTAConfig(n_functions=3, epochs=1)
    def dummy_eval_func(D, h): return torch.matmul(D,h)
    manager.init_sparse_coding_per_block(config=sc_config, evaluation_func=dummy_eval_func)

    X_test = torch.tensor([
        [-1.5, -0.5], # Para bloque (0,0) conceptual: x,y en [-2,0]
        [-0.5,  1.5], # Para bloque (0,1) conceptual: x en [-2,0], y en [0,2]
        [ 0.5, -1.5], # Para bloque (1,0) conceptual: x en [0,2], y en [-2,0]
        [ 1.5,  0.5]  # Para bloque (1,1) conceptual: x,y en [0,2]
    ], device=device, dtype=torch.float32)

    test_active_blocks = manager.retrieve_inference_blocks(X_test)
    assert len(test_active_blocks) == 4

    expected_conceptual_origins_and_points = {
        (0,0): {'origin': torch.tensor([-2.0, -2.0], device=device), 'point': X_test[0]}, # Esperado norm: ([-1.5,-0.5] - [-2,-2])/[2,2] = [0.5,1.5]/[2,2] = [0.25, 0.75]
        (0,1): {'origin': torch.tensor([-2.0,  0.0], device=device), 'point': X_test[1]}, # Esperado norm: ([-0.5,1.5] - [-2,0])/[2,2] = [1.5,1.5]/[2,2] = [0.75, 0.75]
        (1,0): {'origin': torch.tensor([ 0.0, -2.0], device=device), 'point': X_test[2]}, # Esperado norm: ([0.5,-1.5] - [0,-2])/[2,2] = [0.5,0.5]/[2,2] = [0.25, 0.25]
        (1,1): {'origin': torch.tensor([ 0.0,  0.0], device=device), 'point': X_test[3]}, # Esperado norm: ([1.5,0.5] - [0,0])/[2,2] = [1.5,0.5]/[2,2] = [0.75, 0.25]
    }
    
    expected_normalized_values = {
        (0,0): torch.tensor([0.25, 0.75], device=device),
        (0,1): torch.tensor([0.75, 0.75], device=device),
        (1,0): torch.tensor([0.25, 0.25], device=device),
        (1,1): torch.tensor([0.75, 0.25], device=device),
    }

    for test_block in test_active_blocks:
        block_idx = test_block.block_index
        data = expected_conceptual_origins_and_points[block_idx]
        original_point_for_this_block = data['point']
        
        # El block_scope del test_block SÍ incluye el eps, ya que se crea como un PartitionBlock nuevo.
        # Pero la normalización DENTRO de ese test_block debe usar el origen conceptual.
        conceptual_origin_of_test_block = test_block.space_origin + torch.mul(
            torch.tensor(test_block.block_index, device=device, dtype=test_block.space_origin.dtype),
            test_block.block_size
        )
        assert torch.allclose(conceptual_origin_of_test_block, data['origin'])
        
        # La normalización esperada se calcula con el origen conceptual
        calculated_expected_normalized_value = (original_point_for_this_block - conceptual_origin_of_test_block) / test_block.block_size
        
        assert torch.allclose(calculated_expected_normalized_value, expected_normalized_values[block_idx], atol=1e-7), \
            f"Error en el cálculo manual del valor normalizado esperado para el bloque {block_idx}"

        assert test_block.normalized_X.get_for_device(device) is not None
        assert torch.allclose(test_block.normalized_X.get_for_device(device)[0], expected_normalized_values[block_idx], atol=1e-7), \
            f"Normalized_X incorrecto para el bloque {block_idx}. Esperado: {expected_normalized_values[block_idx]}, Obtenido: {test_block.normalized_X[0]}"

        assert torch.all(test_block.normalized_X.get_for_device(device)[0] >= 0.0) and torch.all(test_block.normalized_X.get_for_device(device)[0] <= 1.0), \
             f"Valor normalizado fuera de [0,1] para el bloque {block_idx}: {test_block.normalized_X.get_for_device(device)[0]}"


if __name__ == "__main__":
    from pytest_helper import print_pytest_instructions
    print_pytest_instructions()

