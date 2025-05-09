'''
Copyright (C) 2023-2025 Tecnológico de Costa Rica

Sparse Coding Factory

Provides a factory to produce sparse coding layers.

Authors: The SESM Team 

License: 
'''

from typing import Type, Dict, Any, Optional, Callable
from pysesm.models.SparseCodingBaseLayer import SparseCodingBaseLayer, SparseCodingBaseConfig

class SparseCodingFactory:
    """
    Factory class for creating instances of sparse coding types.
    
    Provides a centralized way to instantiate different variants of sparse coding algorithms
    (classic ISTA, FISTA, etc.) with consistent parameter handling.
    """
    
    _registered_solvers: Dict[str, Type[SparseCodingBaseLayer]] = {}

    @classmethod
    def register(cls, solver_id: str, solver_class: Type[SparseCodingBaseLayer] = None):
        """
        Register a solver implementation with the factory.
        
        Can be used as a decorator or called directly:
        @SparseCodingFactory.register("my_solver")
        class MySolver(SparseCodingBaseLayer):
            pass
            
        OR
        
        SparseCodingFactory.register("my_solver", MySolver)
        """

        def _register(solver_class):
            cls._registered_solvers[solver_id] = solver_class
            return solver_class
            
        if solver_class is None:
            # Used as decorator
            return _register
        
        # Used as direct call
        _register(solver_class)
        return solver_class
    
    @classmethod
    def create(cls, 
               solver_id: str, 
               config: SparseCodingBaseConfig,
               logger: Optional[logging.Logger] = None,
               debug: bool = False,
               parameter_hook: Optional[Callable] = None,
               device: Optional[torch.device] = None) -> SparseCodingBaseLayer:
        """Create an instance of the specified sparse coder type."""
        
        if solver_id not in cls._registered_solvers:
            raise ValueError(f"Unknown sparse coding type: {solver_id}. "
                             f"Available types: {list(cls._registered_solvers.keys())}")
            
        solver_class = cls._registered_solvers[solver_id]
        
        # Validate config is of the correct type
        if not isinstance(config, solver_class.CONFIG_CLASS):
            raise TypeError(f"Solver '{solver_id}' requires config of type "
                            f"{solver_class.CONFIG_CLASS.__name__}, got {type(config).__name__}")
            
        # Create the solver with just the config
        return solver_class(config=config, 
                            logger=logger, 
                            debug=debug,
                            parameter_hook=parameter_hook,
                            device=device)
        
    @classmethod
    def get_available_solvers(cls):
        """Return a list of all registered solver IDs"""
        return list(cls._registered_solvers.keys())
