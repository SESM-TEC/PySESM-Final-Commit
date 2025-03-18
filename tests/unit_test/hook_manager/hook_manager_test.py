import unittest
from unittest.mock import MagicMock, patch
from pysesm.hooks_manager.HookManager import HookManager, HookType

def test_default_initialization():
    """Test that HookManager initializes with default values."""
    logger = MagicMock()
    manager = HookManager(logger=logger)
    
    # Verify default values
    assert not manager.use_wandb
    assert manager.project_name != "sesm-project"
    assert not manager.active_hooks[HookType.ISTALAYER]
    assert not manager.active_hooks[HookType.DICTLAYER]
    assert len(manager.local_storage[HookType.ISTALAYER]) == 0
    assert len(manager.local_storage[HookType.DICTLAYER]) == 0

def test_activate_hooks():
    """Test that hooks can be activated correctly."""
    logger = MagicMock()
    manager = HookManager(logger=logger)
    
    # Activate ISTA hook
    manager.activate_hooks([HookType.ISTALAYER])
    assert manager.active_hooks[HookType.ISTALAYER]
    assert not manager.active_hooks[HookType.DICTLAYER]

    # Activate DICTLAYER hook
    manager.activate_hooks([HookType.DICTLAYER])
    assert manager.active_hooks[HookType.DICTLAYER]

def test_log_hook_data_without_wandb():
    """Test logging hook data without W&B."""
    logger = MagicMock()
    manager = HookManager(logger=logger)
    
    # Activate ISTA hook
    manager.activate_hooks([HookType.ISTALAYER])
    
    # Log data for ISTA hook
    test_data = {"epoch": 1, "loss": 0.5}
    manager.log_hook_data(HookType.ISTALAYER, test_data)
    
    # Verify data is stored locally
    assert len(manager.local_storage[HookType.ISTALAYER]) == 1
    assert manager.local_storage[HookType.ISTALAYER][0] == test_data

def test_log_hook_data_with_wandb():
    """Test logging hook data with W&B."""
    logger = MagicMock()
    manager = HookManager(use_wandb=True, logger=logger)
    
    # Mock wandb.log to avoid actual W&B calls
    with patch("wandb.log") as mock_wandb_log:
        # Activate DICTLAYER hook
        manager.activate_hooks([HookType.DICTLAYER])
        
        # Log data for DICTLAYER hook
        test_data = {"epoch": 1, "loss": 0.5}
        manager.log_hook_data(HookType.DICTLAYER, test_data)
        
        # Verify wandb.log was called
        mock_wandb_log.assert_called_once_with(test_data)

def test_log_hook_data_inactive_hook():
    """Test logging data for an inactive hook."""
    logger = MagicMock()
    manager = HookManager(logger=logger)
    
    # Log data for ISTA hook (inactive)
    test_data = {"epoch": 1, "loss": 0.5}
    manager.log_hook_data(HookType.ISTALAYER, test_data)
    
    # Verify no data is stored
    assert len(manager.local_storage[HookType.ISTALAYER]) == 0

def test_wandb_initialization_failure():
    """Test that W&B initialization failure is handled gracefully."""
    logger = MagicMock()
    
    # Mock wandb.init to raise an exception
    with patch("wandb.init", side_effect=Exception("W&B failed")):
        manager = HookManager(use_wandb=True, logger=logger)
        
        # Verify W&B is disabled after initialization failure
        assert not manager.use_wandb
        logger.error.assert_called_once_with("Failed to initialize W&B: W&B failed")

def test_project_name_customization():
    """Test that the project name can be customized."""
    logger = MagicMock()
    custom_project_name = "custom-project"
    manager = HookManager(use_wandb=True, project_name=custom_project_name, logger=logger)
    
    # Verify custom project name is set
    assert manager.project_name == custom_project_name

if __name__ == "__main__":
    from pytest_helper import print_pytest_instructions
    print_pytest_instructions()
