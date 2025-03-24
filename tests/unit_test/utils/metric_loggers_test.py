import wandb
from unittest.mock import MagicMock, patch
from pysesm.utils.metric_loggers import generic_hook

def test_console_output(capsys):
    """Test that console output works correctly"""
    layer_name = "TestLayer"
    info = {"loss": 0.5, "accuracy": 0.9}
    
    generic_hook(layer_name, info)
    
    captured = capsys.readouterr()
    expected_message = f"{layer_name} - loss: 0.5, accuracy: 0.9"
    assert captured.out.strip() == expected_message

def test_logger_output():
    """Test that logger is called when provided"""
    logger = MagicMock()
    layer_name = "TestLayer"
    info = {"loss": 0.5, "accuracy": 0.9}
    
    generic_hook(layer_name, info, logger=logger)
    
    expected_message = f"{layer_name} - loss: 0.5, accuracy: 0.9"
    logger.info.assert_called_once_with(expected_message)

def test_no_logger_output():
    """Test that no logger call happens when logger is None"""
    logger = MagicMock()
    layer_name = "TestLayer"
    info = {"loss": 0.5, "accuracy": 0.9}
    
    generic_hook(layer_name, info, logger=None)
    logger.info.assert_not_called()

@patch('wandb.init')
@patch('wandb.log')
def test_wandb_initialization(mock_wandb_log, mock_wandb_init):
    """Test W&B initialization and logging when use_wandb=True"""
    layer_name = "TestLayer"
    info = {"loss": 0.5, "accuracy": 0.9}
    project_name = "test_project"
    wandb.run = None  # Simulate no active W&B run
    
    generic_hook(layer_name, info, project_name=project_name, use_wandb=True)
    
    mock_wandb_init.assert_called_once_with(project=project_name)
    expected_log = {
        f"{layer_name}/loss": 0.5,
        f"{layer_name}/accuracy": 0.9
    }
    mock_wandb_log.assert_called_once_with(expected_log)

@patch('wandb.init')
@patch('wandb.log')
def test_wandb_no_reinit(mock_wandb_log, mock_wandb_init):
    """Test W&B doesn't reinitialize if run exists"""
    layer_name = "TestLayer"
    info = {"loss": 0.5, "accuracy": 0.9}
    project_name = "test_project"
    wandb.run = MagicMock()  # Simulate active W&B run
    
    generic_hook(layer_name, info, project_name=project_name, use_wandb=True)
    
    mock_wandb_init.assert_not_called()
    mock_wandb_log.assert_called_once()

@patch('wandb.init')
@patch('wandb.log')
def test_no_wandb_output(mock_wandb_log, mock_wandb_init):
    """Test no W&B calls when use_wandb=False"""
    layer_name = "TestLayer"
    info = {"loss": 0.5, "accuracy": 0.9}
    project_name = "test_project"
    
    generic_hook(layer_name, info, project_name=project_name, use_wandb=False)
    
    mock_wandb_init.assert_not_called()
    mock_wandb_log.assert_not_called()

def test_empty_info_dict(capsys):
    """Test behavior with empty info dictionary"""
    logger = MagicMock()
    layer_name = "TestLayer"
    
    generic_hook(layer_name, {}, logger=logger)
    
    # Check console output
    captured = capsys.readouterr()
    assert captured.out.strip() != f"{layer_name} - "
    
    # Check logger output
    logger.info.assert_called_once_with(f"{layer_name} - ")

def test_none_values_in_info(capsys):
    """Test handling of None values in info dict"""
    layer_name = "TestLayer"
    info = {"loss": None, "accuracy": 0.9}
    
    generic_hook(layer_name, info)
    
    captured = capsys.readouterr()
    assert captured.out.strip() == f"{layer_name} - loss: None, accuracy: 0.9"


if __name__ == "__main__":
    from ..pytest_helper import print_pytest_instructions
    print_pytest_instructions()