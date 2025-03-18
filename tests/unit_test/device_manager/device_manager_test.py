import torch
import logging
import pytest
from unittest.mock import MagicMock, patch
from pysesm.enums.DeviceTargetEnum import DeviceTarget
from pysesm.device_manager.DeviceManager import DeviceManager
def test_default_device_cpu():
    """Test that DeviceManager defaults to CPU."""
    logger = MagicMock()
    manager = DeviceManager(logger=logger)
    assert manager.default_device == "cpu"

@patch("torch.cuda.is_available", return_value=False)
def test_cuda_unavailable_raises_error(mock_cuda_available):
    """Test that requesting CUDA when unavailable raises an error."""
    logger = MagicMock()
    with pytest.raises(RuntimeError):
        DeviceManager(logger=logger, device_map={DeviceTarget.GLOBAL: "cuda:0"})

@patch("torch.cuda.is_available", return_value=True)
@patch("torch.cuda.device_count", return_value=2)
def test_valid_cuda_device(mock_cuda_available, mock_device_count):
    """Test that a valid CUDA device is assigned correctly."""
    logger = MagicMock()
    manager = DeviceManager(logger=logger, device_map={DeviceTarget.GLOBAL: "cuda:1"})
    assert manager.get_device(DeviceTarget.GLOBAL) == "cuda:1"

@patch("torch.cuda.is_available", return_value=True)
@patch("torch.cuda.device_count", return_value=1)
def test_invalid_cuda_device_index(mock_cuda_available, mock_device_count):
    """Test that an invalid CUDA device index raises an error."""
    logger = MagicMock()
    with pytest.raises(RuntimeError):
        DeviceManager(logger=logger, device_map={DeviceTarget.GLOBAL: "cuda:2"})

def test_get_device_with_default():
    """Test getting device assignments with a default fallback."""
    logger = MagicMock()
    manager = DeviceManager(logger=logger, device_map={DeviceTarget.ISTA_LAYER: "cpu"})
    assert manager.get_device(DeviceTarget.ISTA_LAYER) == "cpu"
    assert manager.get_device("custom_component") != "cuda:0"  #c Default fallback


if __name__ == "__main__":
    from pytest_helper import print_pytest_instructions
    print_pytest_instructions()
