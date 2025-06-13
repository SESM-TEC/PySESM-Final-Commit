# tests/unit_test/sparse_coding/base_sparse_coding_test.py
import pytest
import torch
import logging
from abc import ABC, abstractmethod
from unittest.mock import MagicMock

# Import base class and config for type hinting
from pysesm.sparse_coding.SparseCodingBaseLayer import SparseCodingBaseLayer, SparseCodingConfig
from pysesm.enums.DeviceTargetEnum import DeviceTarget # For device manager tests

# This file defines the base tests that all specific sparse coding layers should pass.
# It focuses on "programming correctness" rather than numerical convergence.

class BaseSparseCodingTest(ABC):
    """
    Abstract base class for testing sparse coding layers (ISTA, FISTA, ADMM).
    Subclasses must implement the `layer_factory` fixture to provide instances
    of the specific layer under test.
    """

    @pytest.fixture(autouse=True)
    @abstractmethod
    def layer_factory(self, common_logger, common_device_manager, common_evaluation_func):
        """
        Abstract factory fixture that concrete test classes must implement.
        It should return an object with:
        - `config_class`: The Config class for the specific layer (e.g., ISTAConfig).
        - `create(config)`: A method to create an instance of the layer.
        """
        # Example implementation in a subclass:
        # class SpecificLayerFactory:
        #     config_class = SpecificLayerConfig
        #     def create(self, config):
        #         return SpecificLayer(
        #             config=config,
        #             evaluation_func=common_evaluation_func,
        #             logger=common_logger,
        #             device=common_device_manager.get_device(DeviceTarget.SPARSE_CODING_LAYER)
        #         )
        # return SpecificLayerFactory()
        pass

    # --- Common Tests Focusing on Programming Correctness ---

    def test_initialization_of_h_tensor_properties(self, layer_factory, common_device_manager):
        """
        Verify that `h` is initialized as a PyTorch Parameter, is on the correct device,
        and has `requires_grad=False`.
        """
        n_functions = 5
        config = layer_factory.config_class(n_functions=n_functions)
        layer = layer_factory.create(config) # setup() is called in __init__
    
        assert isinstance(layer.h, torch.nn.Parameter), "`h` should be a torch.nn.Parameter"
        assert layer.h.requires_grad is False, "`h` should not require gradients in sparse coding"
        # CORRECTED: Convert torch.device object to string for comparison
        assert str(layer.h.device) == common_device_manager.get_device(DeviceTarget.SPARSE_CODING_LAYER), \
            f"`h` should be on the correct device. Expected {common_device_manager.get_device(DeviceTarget.SPARSE_CODING_LAYER)}, got {layer.h.device}"
        assert layer.h.shape == (n_functions, 1), "`h` should have shape (n_functions, 1)"

    def test_setup_with_custom_h(self, layer_factory, common_device_manager):
        """
        Verify that `setup()` correctly initializes `h` with a custom tensor if provided.
        """
        n_functions = 5
        custom_h_val = torch.randn(n_functions, 1)
        
        # Test 1: Initialize config with initial_h
        config = layer_factory.config_class(n_functions=n_functions, initial_h=custom_h_val)
        layer = layer_factory.create(config)
        assert torch.allclose(layer.h.cpu(), custom_h_val), "Layer should initialize h with provided `initial_h` from config"
        assert layer.h.requires_grad is False # Still no gradients

        # Test 2: Call setup() directly with a custom h
        new_custom_h_val = torch.ones(n_functions, 1)
        layer.setup(new_custom_h_val)
        assert torch.allclose(layer.h.cpu(), new_custom_h_val), "setup() should correctly update h with a new custom value"
        assert layer.h.requires_grad is False

        # Test 3: Providing 1D tensor for h, should be reshaped to 2D
        one_d_h = torch.randn(n_functions)
        layer.setup(one_d_h)
        assert layer.h.shape == (n_functions, 1), "h should be reshaped to (n_functions, 1)"
        assert torch.allclose(layer.h.cpu().squeeze(), one_d_h), "Reshaped h should contain correct values"

        # Test 4: Mismatching dimensions for h should raise ValueError
        with pytest.raises(ValueError, match="Dimension mismatch"):
            layer.setup(torch.randn(n_functions + 1, 1))
        
    def test_losses_list_appended_per_epoch(self, layer_factory, sparse_coding_data_generator):
        """
        Verify that the `losses` list is correctly appended during `partial_fit`
        and contains one entry per epoch.
        """
        n_samples = 10
        n_features = 2
        n_functions = 3
        epochs = 10
        
        dictionary_D, _, target_y = sparse_coding_data_generator(n_samples, n_features, n_functions)
        config = layer_factory.config_class(epochs=epochs, n_functions=n_functions)
        layer = layer_factory.create(config)
        
        layer.partial_fit(target_y, dictionary_D)
        
        assert len(layer.losses) == epochs, "Losses list should have an entry for each epoch"
        assert all(isinstance(loss, float) for loss in layer.losses), "Loss entries should be floats"
        
    def test_parameter_hook_called_correctly(self, layer_factory, sparse_coding_data_generator):
        """
        Verify that the `parameter_hook` is called with the expected information
        during each training step.
        """
        mock_hook = MagicMock()
        n_samples = 10
        n_features = 2
        n_functions = 3
        epochs = 3
        
        dictionary_D, _, target_y = sparse_coding_data_generator(n_samples, n_features, n_functions)
        config = layer_factory.config_class(epochs=epochs, n_functions=n_functions)
        layer = layer_factory.create(config, parameter_hook=mock_hook)
        
        layer.partial_fit(target_y, dictionary_D)
        
        assert mock_hook.call_count == epochs, "Parameter hook should be called once per epoch"
        
        # Verify content of the calls
        for call_args, _ in mock_hook.call_args_list:
            info_dict = call_args[0]
            assert 'h' in info_dict, "Hook info should contain 'h'"
            assert 'loss' in info_dict, "Hook info should contain 'loss'"
            assert isinstance(info_dict['h'], torch.Tensor), "'h' in hook info should be a tensor"
            assert info_dict['h'].requires_grad is False, "'h' in hook info should be detached"
            assert isinstance(info_dict['loss'], float), "'loss' in hook info should be a float"

    def test_device_placement_during_training(self, layer_factory, sparse_coding_data_generator, common_device_manager):
        """
        Verify that tensors are moved to the correct device during training steps.
        """
        n_samples = 10
        n_features = 2
        n_functions = 3
        epochs = 1
        
        # Data created on the device specified by DeviceManager
        dictionary_D, _, target_y = sparse_coding_data_generator(n_samples, n_features, n_functions)
        
        config = layer_factory.config_class(epochs=epochs, n_functions=n_functions)
        layer = layer_factory.create(config)
        
        # Ensure layer itself is on the correct device
        assert str(next(layer.parameters()).device) == common_device_manager.get_device(DeviceTarget.SPARSE_CODING_LAYER)
        assert str(layer.h.device) == common_device_manager.get_device(DeviceTarget.SPARSE_CODING_LAYER)

        # Mock the `forward` method of the actual criterion object to observe its inputs
        # The criterion is already initialized by the layer's constructor (e.g., as MSELoss)
        original_criterion_forward = layer.criterion.forward # Store original to restore later if needed
        mock_criterion_forward = MagicMock(side_effect=lambda x, y: torch.tensor(0.0, device=x.device))
        layer.criterion.forward = mock_criterion_forward # Mock only the forward method
                
        layer.train_step(target_y, dictionary_D) # Perform one step
        
        # Verify inputs to criterion are on the correct device
        called_y_pred, called_target_y = mock_criterion_forward.call_args[0]
        assert str(called_y_pred.device) == common_device_manager.get_device(DeviceTarget.SPARSE_CODING_LAYER), \
            "Predicted y in criterion should be on correct device"
        assert str(called_target_y.device) == common_device_manager.get_device(DeviceTarget.SPARSE_CODING_LAYER), \
            "Target y in criterion should be on correct device"

        # Dynamically check type to avoid import cycle for ADMMLayer and FISTALayer
        # Use importlib to get the class if it exists (e.g., if this test is run directly on ADMM/FISTA files)
        try:
            from pysesm.sparse_coding.ADMMLayer import ADMMLayer
            from pysesm.sparse_coding.FISTALayer import FISTALayer
        except ImportError:
            ADMMLayer = FISTALayer = type(None) # Use a dummy type if not available
        
        if isinstance(layer, (ADMMLayer, FISTALayer)):
            assert str(layer.z.device) == common_device_manager.get_device(DeviceTarget.SPARSE_CODING_LAYER), \
                "Auxiliary variable `z` should be on correct device"
            if isinstance(layer, ADMMLayer): # ADMM specific
                assert str(layer.u.device) == common_device_manager.get_device(DeviceTarget.SPARSE_CODING_LAYER), \
                    "Auxiliary variable `u` should be on correct device"

    def test_criterion_is_used(self, layer_factory, sparse_coding_data_generator):
        """
        Verify that the configured criterion (loss function) is actually used
        during the training step.
        """
        n_samples = 10
        n_features = 2
        n_functions = 3
        
        dictionary_D, _, target_y = sparse_coding_data_generator(n_samples, n_features, n_functions)
        
        # Create layer with default criterion (e.g., MSELoss)
        config = layer_factory.config_class(epochs=1, n_functions=n_functions)
        layer = layer_factory.create(config)
        
        # Now, mock the forward method of the criterion instance that was created
        mock_criterion_forward = MagicMock(return_value=torch.tensor(0.123))
        layer.criterion.forward = mock_criterion_forward
        
        layer.train_step(target_y, dictionary_D)
        
        # Assert that the mocked forward method was called
        mock_criterion_forward.assert_called_once()
        # Assert that the loss recorded is the value returned by the mock
        assert layer.losses[-1] == pytest.approx(0.123) # Check if the mock return value was used
        
    def test_forward_pass_does_not_update_h(self, layer_factory, sparse_coding_data_generator):
        """
        Verify that the `forward()` method (prediction without training) does not
        alter the `h` tensor.
        """
        n_samples = 10
        n_features = 2
        n_functions = 3
        
        dictionary_D, _, target_y = sparse_coding_data_generator(n_samples, n_features, n_functions)
        config = layer_factory.config_class(n_functions=n_functions)
        layer = layer_factory.create(config)
        
        # Set h to a known value
        initial_h_val = torch.ones_like(layer.h.data)
        layer.h.data = initial_h_val
        
        layer.forward(target_y, dictionary_D) # Call forward pass
        
        assert torch.allclose(layer.h.data, initial_h_val), "h should not change during forward pass"
        assert layer.h.requires_grad is False # Should still be detached

    def test_initial_h_dimension_handling(self, layer_factory):
        """
        Test that initial_h is correctly handled whether it's 1D or 2D.
        """
        n_functions = 10
        
        # Test with 1D initial_h
        initial_h_1d = torch.randn(n_functions)
        config_1d = layer_factory.config_class(n_functions=n_functions, initial_h=initial_h_1d)
        layer_1d = layer_factory.create(config_1d)
        assert layer_1d.h.shape == (n_functions, 1)
        assert torch.allclose(layer_1d.h.cpu().squeeze(), initial_h_1d)
        
        # Test with 2D initial_h (column vector)
        initial_h_2d = torch.randn(n_functions, 1)
        config_2d = layer_factory.config_class(n_functions=n_functions, initial_h=initial_h_2d)
        layer_2d = layer_factory.create(config_2d)
        assert layer_2d.h.shape == (n_functions, 1)
        assert torch.allclose(layer_2d.h.cpu(), initial_h_2d)

        # Test with 0 functions, should still initialize correctly without error
        config_zero_func = layer_factory.config_class(n_functions=0)
        layer_zero_func = layer_factory.create(config_zero_func)
        assert layer_zero_func.h.shape == (0, 1)
        assert len(layer_zero_func.losses) == 0

    @pytest.fixture(scope="session", autouse=True)
    def register_helper_for_dynamic_imports(self):
        """
        Fixture to register a helper function for dynamic imports,
        needed because ADMMLayer and FISTALayer are not directly imported
        to avoid circular dependencies in the base test file.
        This is a workaround specific to Python's import system if direct imports cause issues.
        """
        # This can remain a placeholder or be removed if not strictly necessary and direct imports are fine.
        pass
