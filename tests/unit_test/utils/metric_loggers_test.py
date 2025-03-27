import wandb
from unittest.mock import MagicMock, patch
from pysesm.utils.metric_loggers import log_to_console, log_to_WB

def test_console_output(capsys):
    """Test that console output works correctly"""
    layer_name = "TestLayer"
    info = {"loss": 0.5, "accuracy": 0.9}
    
    log_to_console(layer_name, info)
    
    captured = capsys.readouterr()
    expected_message = f"{layer_name} - loss: 0.5, accuracy: 0.9"
    assert captured.out.strip() == expected_message

def test_logger_output():
    """Test that logger is called when provided"""
    logger = MagicMock()
    layer_name = "TestLayer"
    info = {"loss": 0.5, "accuracy": 0.9}
    
    log_to_WB(layer_name, info, logger=logger)
    
    expected_message = f"{layer_name} - loss: 0.5, accuracy: 0.9"
    logger.info.assert_called_once_with(expected_message)

def test_no_logger_output():
    """Test that no logger call happens when logger is None"""
    logger = MagicMock()
    layer_name = "TestLayer"
    info = {"loss": 0.5, "accuracy": 0.9}
    
    log_to_WB(layer_name, info, logger=None)
    logger.info.assert_not_called()

@patch('wandb.init')
@patch('wandb.log')
def test_wandb_initialization(mock_wandb_log, mock_wandb_init):
    """Test W&B initialization and logging"""
    layer_name = "TestLayer"
    info = {"loss": 0.5, "accuracy": 0.9}
    project_name = "test_project"
    wandb.run = None  # Simulate no active W&B run
    
    log_to_WB(layer_name, info, project_name=project_name)
    
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
    
    log_to_WB(layer_name, info, project_name=project_name)
    
    mock_wandb_init.assert_not_called()
    mock_wandb_log.assert_called_once()

@patch('wandb.init')
@patch('wandb.log')
def test_wandb_no_project_name(mock_wandb_log, mock_wandb_init):
    """Test W&B initialization without project name"""
    layer_name = "TestLayer"
    info = {"loss": 0.5, "accuracy": 0.9}
    wandb.run = None
    
    log_to_WB(layer_name, info, project_name=None)
    
    mock_wandb_init.assert_called_once_with(project=None)
    mock_wandb_log.assert_called_once()

def test_empty_info_dict(capsys):
    """Test behavior with empty info dictionary"""
    logger = MagicMock()
    layer_name = "TestLayer"
    
    # Test console logging
    log_to_console(layer_name, {})
    captured = capsys.readouterr()
    assert captured.out.strip() == f"{layer_name} -"
    

if __name__ == "__main__":
    from ..pytest_helper import print_pytest_instructions
    print_pytest_instructions()