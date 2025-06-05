# tests/unit_test/sparse_coding/ista_layer_test.py
import pytest
import torch
import logging

# Import the specific layer and config
from pysesm.sparse_coding import ISTALayer, ISTAConfig, StepSizeMethod

# Import the base test class
from base_sparse_coding_test import BaseSparseCodingTest
from pysesm.enums.DeviceTargetEnum import DeviceTarget # For device manager

# Import the helper for __main__ block
from pytest_helper import print_pytest_instructions # CORRECTED IMPORT

# --- Concrete Tests for ISTALayer ---

class TestISTALayer(BaseSparseCodingTest):
    """
    Concrete tests for ISTALayer, inheriting common programming correctness tests
    from BaseSparseCodingTest.
    """
    
    @pytest.fixture
    def layer_factory(self, common_logger, common_device_manager, common_evaluation_func):
        """
        Factory for ISTALayer instances. This fixture provides the specific
        implementation details needed by the abstract BaseSparseCodingTest.
        """
        class ISTALayerFactory:
            config_class = ISTAConfig
            def create(self, config):
                return ISTALayer(
                    config=config,
                    evaluation_func=common_evaluation_func,
                    logger=common_logger,
                    device=common_device_manager.get_device(DeviceTarget.SPARSE_CODING_LAYER)
                )
        return ISTALayerFactory()

    # --- ISTA-specific Programming Stress Tests ---

    def test_ista_h_initialization_strategy(self, layer_factory):
        """
        Verify ISTA's specific `h` initialization: one 1.0, rest 0.0, if n_functions > 0.
        This tests the custom initialization logic within ISTALayer.setup().
        """
        n_functions_test = 5
        config = layer_factory.config_class(n_functions=n_functions_test)
        layer = layer_factory.create(config)

        # Check if exactly one element is 1.0 and others are 0.0
        # This assumes the specific initialization logic in ISTALayer's setup()
        assert torch.sum(layer.h == 1.0).item() == 1, \
            "Expected exactly one element of h to be 1.0 for ISTA default init"
        assert torch.sum(layer.h == 0.0).item() == (n_functions_test - 1), \
            "Expected remaining elements of h to be 0.0 for ISTA default init"
        assert layer.h.requires_grad is False # Still no gradients

    def test_ista_h_update_logic_simple_case(self, layer_factory, sparse_coding_data_generator):
        """
        A simplified test to ensure the `h` update logic (gradient step + soft thresholding)
        is executed without errors and alters `h` as expected for a single step.
        This does not test numerical convergence, but operation flow.
        """
        n_samples = 10
        n_features = 2
        n_functions = 3
        
        dictionary_D, _, target_y = sparse_coding_data_generator(n_samples, n_features, n_functions)
        
        # Configure with a manual alpha for predictable step size
        config = layer_factory.config_class(
            epochs=1, # Only one epoch
            alpha=0.1, # Manual step size
            lambd=0.01, # Small lambda for mild shrinkage
            step_size_method=StepSizeMethod.MANUAL,
            n_functions=n_functions
        )
        layer = layer_factory.create(config)
        
        initial_h = layer.h.data.clone() # Store initial h
        
        layer.train_step(target_y, dictionary_D) # Perform one step
        
        # Verify h has changed
        assert not torch.allclose(layer.h.data, initial_h), "h should have been updated after one train step"
        assert layer.h.requires_grad is False, "`h` should remain requires_grad=False after update"


if __name__ == "__main__":
    from pytest_helper import print_pytest_instructions
    print_pytest_instructions()
