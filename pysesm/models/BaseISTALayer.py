from abc import ABC, abstractmethod
import torch

class BaseISTALayer(torch.nn.Module, ABC):
    """
    ---
    """
    
    @abstractmethod
    def __init__(self, n_functions: int, alpha: float, lambd: float, 
                 evaluation_func: callable, logger, **kwargs):
        super().__init__()
        self.n_functions = n_functions
        self.alpha = alpha
        self.lambd = lambd
        self.evaluation_func = evaluation_func
        self.logger = logger
        
    @abstractmethod
    def setup(self, h: torch.Tensor = None) -> None:
        """Inicializa el vector disperso h"""
        pass
    
    @abstractmethod
    def forward(self, y: torch.Tensor, dictionary: torch.Tensor, 
                log_losses: bool = True) -> torch.Tensor:
        """Realiza el forward pass"""
        pass
    
    @abstractmethod
    def train_step(self, y: torch.Tensor, dictionary: torch.Tensor, 
                   log_losses: bool = True) -> torch.Tensor:
        """Realiza un paso de entrenamiento completo"""
        pass
    
    @abstractmethod
    def partial_fit(self, y: torch.Tensor, epochs: int, 
                    dictionary: torch.Tensor, log_losses: bool = True) -> None:
        """Entrenamiento parcial por lotes"""
        pass
    
    @abstractmethod
    def shrinkage(self) -> torch.Tensor:
        """Operación de shrinkage para promover dispersión"""
        pass