'''
Copyright (C) 2023-2025 Tecnológico de Costa Rica

Sparse Coding Factory

Provides a factory to produce sparse coding layers.

Authors: The SESM Team 

License: 
'''

from typing import Dict, Type, Optional, Callable, TypeVar, Generic, Union, Any
import torch
import logging
from abc import ABC, abstractmethod

from pysesm.models.SparseCodingBaseLayer import SparseCodingBaseLayer, SparseCodingConfig

class SparseCodingFactory:
    """
    Factory class for creating sparse coding solver instances.
    
    This factory provides a centralized way to create different sparse coding solvers
    with flexible registration and instantiation methods. Solvers can be created either
    by a string identifier or directly from a configuration object.
    
    Usage examples:
    
    # Registration (in solver implementation file):
    @SparseCodingFactory.register("classic_ista")
    class ISTALayer(SparseCodingBaseLayer):
        CONFIG_CLASS = ISTAConfig
        # ...implementation...
    
    # Creation by ID:
    solver = SparseCodingFactory.create("classic_ista", config=ista_config)
    
    # Creation by config type:
    solver = SparseCodingFactory.create(ista_config)
    """
    
    _registered_solvers_by_id: Dict[str, Type[SparseCodingBaseLayer]] = {}
    _registered_solvers_by_config: Dict[Type[SparseCodingConfig], Type[SparseCodingBaseLayer]] = {}
    
    @classmethod
    def register(cls, solver_id: str, solver_class: Type[SparseCodingBaseLayer] = None):
        """
        Register a solver implementation with the factory.
        
        This method can be used either as a decorator or called directly:
        
        # As a decorator:
        @SparseCodingFactory.register("classic_ista")
        class ISTALayer(SparseCodingBaseLayer):
            CONFIG_CLASS = ISTAConfig
            # ...implementation...
            
        # Direct call:
        SparseCodingFactory.register("classic_ista", ISTALayer)
        
        Important: Every solver class MUST define a class variable CONFIG_CLASS 
        that specifies which configuration class it expects.
        
        Args:
            solver_id: String identifier for the solver type
            solver_class: The solver class to register (can be None if used as decorator)
            
        Returns:
            The solver class (allowing use as a decorator)
        """
        def _register(solver_class):
            # Verify the solver class has defined CONFIG_CLASS
            if not hasattr(solver_class, 'CONFIG_CLASS'):
                raise AttributeError(
                    f"Solver class {solver_class.__name__} must define a CONFIG_CLASS "
                    f"class variable specifying its configuration type."
                )
            
            # Register by string ID
            cls._registered_solvers_by_id[solver_id] = solver_class
            
            # Also register by config class
            config_class = solver_class.CONFIG_CLASS
            cls._registered_solvers_by_config[config_class] = solver_class
            
            return solver_class
            
        if solver_class is None:
            # Used as decorator
            return _register
        
        # Used as direct call
        return _register(solver_class)
    
    @classmethod
    def create_from_id(cls, 
              solver_id: str, 
              config: SparseCodingConfig,
              logger: Optional[logging.Logger] = None,
              debug: bool = False,
              parameter_hook: Optional[Callable[[dict], None]] = None,
              device: Optional[torch.device] = None) -> SparseCodingBaseLayer:
        """
        Create a solver instance by its registered string identifier.
        
        Args:
            solver_id: String identifier for the registered solver type
            config: Configuration object (must match the solver's expected config type)
            logger: Optional logger for the solver
            debug: Enable debug output
            parameter_hook: Optional callback for parameter updates
            device: Device for computation (CPU/GPU)
            
        Returns:
            An instance of the requested sparse coding solver
            
        Raises:
            ValueError: If no solver is registered with the given ID
            TypeError: If the provided config is not of the expected type
        """
        if solver_id not in cls._registered_solvers_by_id:
            raise ValueError(f"Unknown solver ID: {solver_id}. "
                            f"Available solvers: {list(cls._registered_solvers_by_id.keys())}")
            
        solver_class = cls._registered_solvers_by_id[solver_id]
        
        # Validate config type
        if not isinstance(config, solver_class.CONFIG_CLASS):
            raise TypeError(f"Solver '{solver_id}' requires config of type "
                           f"{solver_class.CONFIG_CLASS.__name__}, got {type(config).__name__}")
        
        # Create solver
        return solver_class(
            config=config, 
            logger=logger, 
            debug=debug,
            parameter_hook=parameter_hook,
            device=device
        )
    
    @classmethod
    def create_from_config(cls,
                          config: SparseCodingConfig,
                          logger: Optional[logging.Logger] = None,
                          debug: bool = False,
                          parameter_hook: Optional[Callable[[dict], None]] = None,
                          device: Optional[torch.device] = None) -> SparseCodingBaseLayer:
        """
        Create a solver instance based on the type of config provided.
        
        This method automatically determines which solver to create based on the
        specific configuration class provided. It's particularly useful when you
        have a configuration object but don't know or care which solver it corresponds to.
        
        Args:
            config: Configuration object that determines which solver to create
            logger: Optional logger for the solver
            debug: Enable debug output
            parameter_hook: Optional callback for parameter updates
            device: Device for computation (CPU/GPU)
            
        Returns:
            An instance of the appropriate sparse coding solver for the config
            
        Raises:
            ValueError: If no solver is registered for the config's type
        """
        config_class = type(config)
        
        if config_class not in cls._registered_solvers_by_config:
            raise ValueError(f"No solver registered for config type: {config_class.__name__}. "
                            f"Available config types: {[c.__name__ for c in cls._registered_solvers_by_config.keys()]}")
            
        solver_class = cls._registered_solvers_by_config[config_class]
        
        # Create solver
        return solver_class(
            config=config, 
            logger=logger, 
            debug=debug,
            parameter_hook=parameter_hook,
            device=device
        )
    
    @classmethod
    def create(cls, *args, **kwargs) -> SparseCodingBaseLayer:
        """
        Flexible creation method that determines which approach to use based on arguments.
        
        This method provides a convenient interface that automatically routes to either
        create_from_id or create_from_config based on the arguments provided.
        
        Usage:
            # Create by ID
            solver = SparseCodingFactory.create("classic_ista", config=ista_config)
            
            # Create by config
            solver = SparseCodingFactory.create(ista_config)
            
            # Also supports keyword arguments
            solver = SparseCodingFactory.create(solver_id="classic_ista", config=ista_config)
            solver = SparseCodingFactory.create(config=ista_config)
        
        Returns:
            An instance of the appropriate sparse coding solver
            
        Raises:
            ValueError: If the method can't determine which creation approach to use
        """
        if len(args) > 0:
            first_arg = args[0]
            if isinstance(first_arg, str):
                return cls.create_from_id(*args, **kwargs)
            #elif isinstance(first_arg, SparseCodingBaseConfig):
            elif isinstance(first_arg, SparseCodingConfig):
                return cls.create_from_config(*args, **kwargs)
                
        # If no positional args or can't determine, check kwargs
        if 'solver_id' in kwargs:
            solver_id = kwargs.pop('solver_id')
            return cls.create_from_id(solver_id, **kwargs)
        elif 'config' in kwargs and isinstance(kwargs['config'], SparseCodingConfig):
            return cls.create_from_config(**kwargs)
            
        raise ValueError(
            "Could not determine creation method. Please provide either:\n"
            "1. A solver ID string and config: create('classic_ista', config)\n"
            "2. Just a config object: create(ista_config)\n"
            "3. Named parameters: create(solver_id='classic_ista', config=config)\n"
            "4. Named config: create(config=ista_config)"
        )
    
    @classmethod
    def get_available_solvers(cls):
        """
        Return a list of all registered solver IDs.
        
        This helper method allows discovering which solvers are available
        at runtime.
        
        Returns:
            List of string identifiers for all registered solvers
        """
        return list(cls._registered_solvers_by_id.keys())
    
    @classmethod
    def get_available_config_types(cls):
        """
        Return a list of all registered configuration types.
        
        This helper method allows discovering which configuration
        classes can be used with the factory.
        
        Returns:
            List of configuration classes for all registered solvers
        """
        return list(cls._registered_solvers_by_config.keys())
