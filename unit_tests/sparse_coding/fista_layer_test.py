"""
FISTA Layer Tests.

Concrete tests for the FISTALayer, focusing on momentum parameter updates,
restart strategies, and auxiliary variable management.

Copyright (c) 2023-2025, Tecnológico de Costa Rica
All rights reserved.

This source code is licensed under the BSD 3-Clause License found in the
LICENSE file in the root directory of this source tree.

SPDX-License-Identifier: BSD-3-Clause
"""

import pytest
import torch
import logging

# Import the specific layer and config
from pysesm.sparse_coding.FISTALayer import FISTALayer, FISTAConfig, RestartStrategy, MomentumScheme, StepSizeMethod

# Import the base test class
from base_sparse_coding_test import BaseSparseCodingTest

# Import the helper for __main__ block
from pytest_helper import print_pytest_instructions # CORRECTED IMPORT

# --- Concrete Tests for FISTALayer ---

class TestFISTALayer(BaseSparseCodingTest):
    """
    Concrete tests for FISTALayer, inheriting common programming correctness tests
    from BaseSparseCodingTest.
    """
    
    @pytest.fixture
    def layer_factory(self, common_logger, common_device, common_evaluation_func):
        """
        Factory for FISTALayer instances. This fixture provides the specific
        implementation details needed by the abstract BaseSparseCodingTest.
        """
        class FISTALayerFactory:
            config_class = FISTAConfig
            def create(self, config,**kwargs):
                return FISTALayer(
                    config=config,
                    evaluation_func=common_evaluation_func,
                    logger=common_logger,
                    **kwargs
                )
        return FISTALayerFactory()

    # --- FISTA-specific Programming Stress Tests ---

    def test_fista_auxiliary_variable_initialization(self, layer_factory):
        """
        Verify that FISTA's auxiliary variable `z` and momentum `t` are correctly initialized.
        `z` should be a copy of `h` and `t` should be 1.0.
        """
        n_functions_test = 5
        config = layer_factory.config_class(n_functions=n_functions_test)
        layer = layer_factory.create(config)

        assert hasattr(layer, 'z'), "`z` auxiliary variable should exist"
        assert torch.allclose(layer.z, layer.h.data), "`z` should be initialized as a copy of `h`'s data"
        assert layer.z.requires_grad is False, "`z` should not require gradients"
        
        assert hasattr(layer, 't'), "`t` momentum parameter should exist"
        assert layer.t == 1.0, "`t` should be initialized to 1.0"
        
        assert hasattr(layer, 'prev_h'), "`prev_h` should exist"
        assert torch.allclose(layer.prev_h, layer.h.data), "`prev_h` should be initialized as a copy of `h`'s data"


    def test_fista_momentum_update_logic(self, layer_factory, sparse_coding_data_generator):
        """
        Verify that FISTA's momentum parameter `t` and auxiliary variable `z` are updated
        correctly over multiple steps according to the chosen scheme (e.g., ORIGINAL).
        """
        n_samples = 10
        n_features = 2
        n_functions = 3
        epochs = 5
        
        dictionary_D, _, target_y = sparse_coding_data_generator(n_samples, n_features, n_functions)
        
        # Configure FISTA with ORIGINAL momentum scheme for predictable updates
        config = layer_factory.config_class(
            epochs=epochs,
            alpha=0.1,
            lambd=0.01,
            step_size_method=StepSizeMethod.MANUAL, # Keep step_size constant for easier verification
            n_functions=n_functions,
            momentum_scheme=MomentumScheme.ORIGINAL,
            restart_strategy=RestartStrategy.NONE # No restarts for this test
        )
        layer = layer_factory.create(config)
        
        # Expected `t` values for ORIGINAL scheme starting from t=1.0:
        # t_0 = 1.0 (initial)
        # t_1 = (1 + sqrt(1 + 4*1^2))/2 = (1 + sqrt(5))/2 approx 1.618
        # t_2 = (1 + sqrt(1 + 4*1.618^2))/2 approx 2.236
        # etc.
        expected_t_values = [1.0]
        current_t = torch.tensor(1.0, dtype=torch.float32)
        for _ in range(epochs):
            current_t = (1.0 + torch.sqrt(1.0 + 4.0 * current_t**2)) / 2.0
            expected_t_values.append(current_t)
        
        # Run partial fit and track `t` in a mock hook
        mock_hook_t_values = []
        def custom_hook(info):
            mock_hook_t_values.append(info['t'])
        
        layer.parameter_hook = custom_hook # Override with our mock hook
        layer.partial_fit(target_y, dictionary_D)
        
        # The hook is called *after* the update, so its first value corresponds to epoch 0, 
        # which means t has already been updated for epoch 1 calculation, etc.
        # Thus, it should reflect the t values *used in the update for the current epoch*.
        # FISTA updates t at the end of the step, so the t in the hook is for the next step's z.
        # So, mock_hook_t_values[0] should be layer.t after 1st epoch, and so on.
        assert len(mock_hook_t_values) == epochs

        # Compare recorded t values with expected (skip initial t=1.0)
        # The first reported t value is for the first step, which is calculated based on initial t=1.0.
        # So compare mock_hook_t_values with expected_t_values[1:] (as expected_t_values[0] is the initial t)
        for i in range(epochs):
            assert pytest.approx(mock_hook_t_values[i], rel=1e-4) == expected_t_values[i+1], \
                f"Momentum t mismatch at epoch {i+1}. Expected {expected_t_values[i+1]:.4f}, got {mock_hook_t_values[i]:.4f}"


if __name__ == "__main__":
    from pytest_helper import print_pytest_instructions
    print_pytest_instructions()
