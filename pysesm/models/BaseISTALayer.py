from abc import ABC, abstractmethod
import torch

class BaseISTALayer(torch.nn.Module, ABC):
    """Abstract base class for ISTA layers.
    
    This class defines the interface that all concrete ISTA layer implementations must follow.
    It inherits from both torch.nn.Module and ABC (Abstract Base Class).
    """

    @abstractmethod
    def __init__(self, n_functions: int, alpha: float, lambd: float,
                 evaluation_func: callable, logger, **kwargs):
        """Initialize the base ISTA layer.
        
        Args:
            n_functions: Number of basis functions/dictionary atoms
            alpha: Step size parameter (learning rate)
            lambd: Regularization parameter (controls sparsity)
            evaluation_func: Callable function for model evaluation
            logger: Logger instance for tracking training progress
            **kwargs: Additional implementation-specific parameters
        """
        super().__init__()
        self.n_functions = n_functions
        self.alpha = alpha
        self.lambd = lambd
        self.evaluation_func = evaluation_func
        self.logger = logger
        
    @abstractmethod
    def setup(self, h: torch.Tensor = None) -> None:
        """Initialize the sparse coefficient vector h.
        
        Args:
            h: Optional initial values for the sparse coefficients. If None,
               the implementation should provide its own initialization.
        """
        pass
    
    @abstractmethod
    def forward(self, y: torch.Tensor, dictionary: torch.Tensor,
                log_losses: bool = True) -> torch.Tensor:
        """Perform the forward pass of the ISTA algorithm.
        
        Args:
            y: Input signal tensor
            dictionary: Dictionary/weight matrix for sparse coding
            log_losses: Whether to log loss values during computation
            
        Returns:
            The sparse code representation of the input signal
        """
        pass
    
    @abstractmethod
    def train_step(self, y: torch.Tensor, dictionary: torch.Tensor,
                   log_losses: bool = True) -> torch.Tensor:
        """Perform a complete training step (forward + backward operations).
        
        Args:
            y: Input signal tensor
            dictionary: Dictionary/weight matrix for sparse coding
            log_losses: Whether to log loss values during computation
            
        Returns:
            The updated sparse code representation
        """
        pass
    
    @abstractmethod
    def partial_fit(self, y: torch.Tensor, epochs: int,
                    dictionary: torch.Tensor, log_losses: bool = True) -> None:
        """Perform partial/mini-batch training.
        
        Args:
            y: Input signal tensor
            epochs: Number of training epochs to run
            dictionary: Dictionary/weight matrix for sparse coding
            log_losses: Whether to log loss values during computation
        """
        pass
    
    @abstractmethod
    def shrinkage(self) -> torch.Tensor:
        """Apply shrinkage operation to promote sparsity.
        
        Returns:
            The thresholded/sparsified coefficients
        """
        pass