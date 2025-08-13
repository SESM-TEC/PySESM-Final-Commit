'''
Copyright (C) 2023-2025 Tecnológico de Costa Rica

Generic Factory Base Class

Provides a reusable factory pattern for creating different types of objects
with registration and configuration-based instantiation.

Authors: The SESM Team 

License: 
'''
from __future__ import annotations

from typing import TypeVar, Generic
from ..base_types import BaseConfig

# Define type variables for the generic factory
T_Product = TypeVar('T_Product')  # The product being created (Layer, etc.)
T_Config = TypeVar('T_Config')    # The configuration type

class GenericFactory(Generic[T_Product, T_Config]):
    """
    Generic factory base class that can be specialized for different product types.
    
    This factory handles registration, creation, and validation logic that is
    common across all factory types (SparseCoding, Dictionary, etc.).
    
    Uses __init_subclass__ to automatically provide each subclass with its own
    registration dictionaries, implementing the monostate pattern cleanly.
    
    Usage:
        class MyFactory(GenericFactory[MyProduct, MyConfig]):
            def __init__(self):
                super().__init__(product_name="my_product", config_name="my_config")
        
        # Register products
        @MyFactory.register("my_id")
        class MyProductImpl(MyProduct):
            CONFIG_CLASS = MyConfig
    """
    
    def __init_subclass__(cls, **kwargs):
        """
        Automatically provide each subclass with its own registration dictionaries.
        
        This ensures that SparseCodingFactory and DictFactory have completely
        separate registrations, implementing the monostate pattern cleanly.
        """
        super().__init_subclass__(**kwargs)
        # Each subclass gets its own registration dictionaries
        cls._registered_by_id = {}
        cls._registered_by_config = {}
    
    def __init__(self, product_name: str, config_name: str):
        """
        Initialize the factory with names for error messages.
        
        Args:
            product_name: Human-readable name for the product type (e.g., "sparse_coding_layer")
            config_name: Human-readable name for the config type (e.g., "sparse coding config")
        """
        self.product_name = product_name
        self.config_name = config_name
    
    @classmethod
    def register(cls, product_id: str, product_class: type[T_Product] = None):
        """
        Register a product implementation with the factory.
        
        Can be used as decorator or direct call:
        
        @MyFactory.register("my_id")
        class MyProductImpl(MyProduct):
            CONFIG_CLASS = MyConfig
        
        Args:
            product_id: String identifier for the product type
            product_class: The product class to register (None if used as decorator)
            
        Returns:
            The product class (allowing use as a decorator)
        """
        def _register(product_class):
            # Verify the product class has defined CONFIG_CLASS
            if not hasattr(product_class, 'CONFIG_CLASS'):
                raise AttributeError(
                    f"Product class {product_class.__name__} must define a CONFIG_CLASS "
                    f"class variable specifying its configuration type."
                )
            
            # Register by string ID
            cls._registered_by_id[product_id] = product_class
            
            # Also register by config class
            config_class = product_class.CONFIG_CLASS
            cls._registered_by_config[config_class] = product_class
            
            return product_class
            
        if product_class is None:
            # Used as decorator
            return _register
        
        # Used as direct call
        return _register(product_class)

    @classmethod
    def create_from_id(cls, product_id: str, config: T_Config, **kwargs) -> T_Product:
        """
        Create a product instance by its registered string identifier.
        
        Args:
            product_id: String identifier for the registered product type
            config: Configuration object (must match the product's expected config type)
            **kwargs: Additional arguments passed to the product constructor
            
        Returns:
            An instance of the requested product
        """
        if not hasattr(cls, '_registered_by_id') or product_id not in cls._registered_by_id:
            available = list(cls._registered_by_id.keys()) if hasattr(cls, '_registered_by_id') else []
            raise ValueError(f"Unknown product ID: {product_id}. Available products: {available}")
            
        product_class = cls._registered_by_id[product_id]
        
        # Validate config type
        if not isinstance(config, product_class.CONFIG_CLASS):
            raise TypeError(f"Product '{product_id}' requires config of type "
                           f"{product_class.CONFIG_CLASS.__name__}, got {type(config).__name__}")
        
        # Create product
        return product_class(config=config, **kwargs)
    
    @classmethod
    def create_from_config(cls, config: T_Config, **kwargs) -> T_Product:
        """
        Create a product instance based on the type of config provided.
        
        Args:
            config: Configuration object that determines which product to create
            **kwargs: Additional arguments passed to the product constructor
            
        Returns:
            An instance of the appropriate product for the config
        """
        config_class = type(config)
        
        if not hasattr(cls, '_registered_by_config') or config_class not in cls._registered_by_config:
            available = list(cls._registered_by_config.keys()) if hasattr(cls, '_registered_by_config') else []
            available_names = [c.__name__ for c in available]
            raise ValueError(f"No product registered for config type: {config_class.__name__}. "
                            f"Available config types: {available_names}")
            
        product_class = cls._registered_by_config[config_class]
        return product_class(config=config, **kwargs)
    
    @classmethod
    def create(cls, *args, **kwargs) -> T_Product:
        """
        Flexible creation method that determines approach based on arguments.
        
        Usage:
            # Create by ID
            product = Factory.create("my_id", config=my_config)
            
            # Create by config
            product = Factory.create(my_config)
        """

        # Let's check if there are any positional arguments
        if len(args) > 0:
            first_arg = args[0] 
            if isinstance(first_arg, str): # is the first positional argument a string?
                return cls.create_from_id(*args, **kwargs)
            # Check if it's a config object
            if isinstance(first_arg, BaseConfig):
                return cls.create_from_config(*args, **kwargs)
                
        # Check kwargs
        if 'product_id' in kwargs:
            product_id = kwargs.pop('product_id')
            return cls.create_from_id(product_id, **kwargs)
        if 'config' in kwargs:
            return cls.create_from_config(**kwargs)
            
        raise ValueError("Could not determine creation method. Provide either product_id and config, or just config.")
    
    @classmethod
    def get_available_products(cls) -> list[str]:
        """Return list of all registered product IDs."""
        if hasattr(cls, '_registered_by_id'):
            return list(cls._registered_by_id.keys())
        return []
    
    @classmethod
    def get_available_config_types(cls) -> list[type[T_Config]]:
        """Return list of all registered configuration types."""
        if hasattr(cls, '_registered_by_config'):
            return list(cls._registered_by_config.keys())
        return []
