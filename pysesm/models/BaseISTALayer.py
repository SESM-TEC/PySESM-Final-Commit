from abc import ABC, abstractmethod
import torch
class BaseISTALayer(torch.nn.Module, ABC):
    """
    Abstract base class for ISTA algorithm implementations.
    Inherits from torch.nn.Module for PyTorch integration and ABC for abstract functionality.
    
    All concrete implementations (ISTALayer, FISTALayer, etc.) must inherit from this class
    and implement the abstract methods.
    """
    
    @abstractmethod
    def __init__(self):
        """
        Base initialization. Child classes must call super().__init__()
        """
        super().__init__()
        
    @abstractmethod
    def setup(self, h: torch.Tensor = None) -> None:
        """
        Initializes the sparse vector h.
        
        Args:
            h (torch.Tensor, optional): Initial value for the sparse vector. 
                   If None, it will be randomly initialized.
        """
        pass
    
    @abstractmethod
    def forward(self, y: torch.Tensor, dictionary: torch.Tensor, 
                log_losses: bool = True) -> torch.Tensor:
        """
        Performs the forward pass.
        
        Args:
            y (torch.Tensor): Target/ground truth vector
            dictionary (torch.Tensor): Dictionary matrix for prediction
            log_losses (bool): If True, logs the computed losses
            
        Returns:
            torch.Tensor: Computed loss
        """
        pass
    
    @abstractmethod
    def train_step(self, y: torch.Tensor, dictionary: torch.Tensor, 
                   log_losses: bool = True) -> torch.Tensor:
        """
        Performs a complete training step (forward + backward + optimization).
        
        Args:
            y (torch.Tensor): Target/ground truth vector
            dictionary (torch.Tensor): Dictionary matrix for prediction
            log_losses (bool): If True, logs the computed losses
            
        Returns:
            torch.Tensor: Computed loss
        """
        pass
    
    @abstractmethod
    def partial_fit(self, y: torch.Tensor, epochs: int, 
                    dictionary: torch.Tensor, log_losses: bool = True) -> None:
        """
        Performs a complete training step (forward + backward + optimization).
        
        Args:
            y (torch.Tensor): Target/ground truth vector
            dictionary (torch.Tensor): Dictionary matrix for prediction
            log_losses (bool): If True, logs the computed losses
            
        Returns:
            torch.Tensor: Computed loss
        """
        pass
    
    @abstractmethod
    def shrinkage(self) -> torch.Tensor:
        """
        Applies shrinkage operation to promote sparsity in parameters.
        
        Returns:
            torch.Tensor: Parameters after applying shrinkage operation
        """
        pass