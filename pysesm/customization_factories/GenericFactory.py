'''
Copyright (C) 2023-2025 Tecnológico de Costa Rica

Generic Factory Base Class

Provides a reusable factory pattern for creating different types of objects
with registration and configuration-based instantiation.

Authors: The SESM Team 

License: 
'''

from typing import Dict, Type, Optional, TypeVar, Generic, Any
from abc import ABC


# Define type variables for the generic factory
T_Product = TypeVar('T_Product')  # The product being created (Layer, etc.)
T_Config = TypeVar('T_Config')    # The configuration type

class GenericFactory(Generic[T_Product, T_Config]):
    """
    Generic factory base class that can be specialized for different product types.
    
    This factory handles registration, creation, and validation logic that is
    common across all factory types (SparseCoding, Dictionary, etc.).
    
    Usage:
        # Create specialized factory
        class MyFactory(GenericFactory[MyProduct, MyConfig]):
            def __init__(self):
                super().__init__(product_name="my_product", config_name="my_config")
        
        # Create singleton instance
        MyFactory = MyFactory()
        
        # Register products
        @MyFactory.register("my_id")
        class MyProductImpl(MyProduct):
            CONFIG_CLASS = MyConfig
    """
    
    def __init__(self, product_name: str, config_name: str):
        """
        Initialize the factory with names for error messages.
        
        Args:
            product_name: Human-readable name for the product type (e.g., "solver", "dictionary")
            config_name: Human-readable name for the config type (e.g., "solver config", "dictionary config")
        """
        self.product_name = product_name
        self.config_name = config_name
        self._registered_by_id: Dict[str, Type[T_Product]] = {}
        self._registered_by_config: Dict[Type[T_Config], Type[T_Product]] = {}
    
    def register(self, product_id: str, product_class: Type[T_Product] = None):
        """
        Register a product implementation with the factory.
        
        This method can be used either as a decorator or called directly:
        
        # As a decorator:
        @factory.register("my_id")
        class MyProductImpl(MyProduct):
            CONFIG_CLASS = MyConfig
            # ...implementation...
            
        # Direct call:
        factory.register("my_id", MyProductImpl)
        
        Important: Every product class MUST define a class variable CONFIG_CLASS 
        that specifies which configuration class it expects.
        
        Args:
            product_id: String identifier for the product type
            product_class: The product class to register (can be None if used as decorator)
            
        Returns:
            The product class (allowing use as a decorator)
        """
        def _register(product_class):
            # Verify the product class has defined CONFIG_CLASS
            if not hasattr(product_class, 'CONFIG_CLASS'):
                raise AttributeError(
                    f"{self.product_name.title()} class {product_class.__name__} must define a CONFIG_CLASS "
                    f"class variable specifying its configuration type."
                )
            
            # Register by string ID
            self._registered_by_id[product_id] = product_class
            
            # Also register by config class
            config_class = product_class.CONFIG_CLASS
            self._registered_by_config[config_class] = product_class
            
            return product_class
            
        if product_class is None:
            # Used as decorator
            return _register
        
        # Used as direct call
        return _register(product_class)
    
    def create_from_id(self, product_id: str, config: T_Config, **kwargs) -> T_Product:
        """
        Create a product instance by its registered string identifier.
        
        Args:
            product_id: String identifier for the registered product type
            config: Configuration object (must match the product's expected config type)
            **kwargs: Additional arguments passed to the product constructor
            
        Returns:
            An instance of the requested product
            
        Raises:
            ValueError: If no product is registered with the given ID
            TypeError: If the provided config is not of the expected type
        """
        if product_id not in self._registered_by_id:
            raise ValueError(f"Unknown {self.product_name} ID: {product_id}. "
                            f"Available {self.product_name}s: {list(self._registered_by_id.keys())}")
            
        product_class = self._registered_by_id[product_id]
        
        # Validate config type
        if not isinstance(config, product_class.CONFIG_CLASS):
            raise TypeError(f"{self.product_name.title()} '{product_id}' requires {self.config_name} of type "
                           f"{product_class.CONFIG_CLASS.__name__}, got {type(config).__name__}")
        
        # Create product
        return product_class(config=config, **kwargs)
    
    def create_from_config(self, config: T_Config, **kwargs) -> T_Product:
        """
        Create a product instance based on the type of config provided.
        
        This method automatically determines which product to create based on the
        specific configuration class provided.
        
        Args:
            config: Configuration object that determines which product to create
            **kwargs: Additional arguments passed to the product constructor
            
        Returns:
            An instance of the appropriate product for the config
            
        Raises:
            ValueError: If no product is registered for the config's type
        """
        config_class = type(config)
        
        if config_class not in self._registered_by_config:
            raise ValueError(f"No {self.product_name} registered for {self.config_name} type: {config_class.__name__}. "
                            f"Available {self.config_name} types: {[c.__name__ for c in self._registered_by_config.keys()]}")
            
        product_class = self._registered_by_config[config_class]
        
        # Create product
        return product_class(config=config, **kwargs)
    
    def create(self, *args, **kwargs) -> T_Product:
        """
        Flexible creation method that determines which approach to use based on arguments.
        
        This method provides a convenient interface that automatically routes to either
        create_from_id or create_from_config based on the arguments provided.
        
        Usage:
            # Create by ID
            product = factory.create("my_id", config=my_config)
            
            # Create by config
            product = factory.create(my_config)
            
            # Also supports keyword arguments
            product = factory.create(product_id="my_id", config=my_config)
            product = factory.create(config=my_config)
        
        Returns:
            An instance of the appropriate product
            
        Raises:
            ValueError: If the method can't determine which creation approach to use
        """
        if len(args) > 0:
            first_arg = args[0]
            if isinstance(first_arg, str):
                return self.create_from_id(*args, **kwargs)
            # Check if it's a config (check for common config attributes)
            elif hasattr(first_arg, '__dataclass_fields__') or hasattr(first_arg, '__dict__'):
                return self.create_from_config(*args, **kwargs)
                
        # If no positional args or can't determine, check kwargs
        id_key = f"{self.product_name}_id"
        if id_key in kwargs:
            product_id = kwargs.pop(id_key)
            return self.create_from_id(product_id, **kwargs)
        elif 'config' in kwargs:
            return self.create_from_config(**kwargs)
            
        raise ValueError(
            f"Could not determine creation method for {self.product_name}. Please provide either:\n"
            f"1. A {self.product_name} ID string and config: create('{self.product_name}_example', config)\n"
            f"2. Just a config object: create({self.config_name}_object)\n"
            f"3. Named parameters: create({self.product_name}_id='{self.product_name}_example', config=config)\n"
            f"4. Named config: create(config={self.config_name}_object)"
        )
    
    def get_available_products(self):
        """
        Return a list of all registered product IDs.
        
        This helper method allows discovering which products are available
        at runtime.
        
        Returns:
            List of string identifiers for all registered products
        """
        return list(self._registered_by_id.keys())
    
    def get_available_config_types(self):
        """
        Return a list of all registered configuration types.
        
        This helper method allows discovering which configuration
        classes can be used with the factory.
        
        Returns:
            List of configuration classes for all registered products
        """
        return list(self._registered_by_config.keys())