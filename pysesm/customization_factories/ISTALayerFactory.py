from pysesm.enums.ISTALayerEnum import ISTALayerEnum
from pysesm.models.BaseISTALayer import BaseISTALayer
from pysesm.models.ISTALayer import ISTALayer

class ISTALayerFactory:
    """Factory for creating ISTALayer instances based on the specified type."""
    
    _layer_map = {
        ISTALayerEnum.CLASSIC: ISTALayer,
        # Register new implementations here:
        # ISTALayerEnum.FISTA: FISTALayer,
        # ISTALayerEnum.ADAPTIVE: AdaptiveISTALayer,
    }
    
    @staticmethod
    def create(kind: ISTALayerEnum, n_functions: int, alpha: float, lambd: float,
               evaluation_func: callable, logger, **kwargs) -> BaseISTALayer:
        """Creates an instance of the specified ISTALayer type.
        
        Args:
            kind: The type of ISTALayer to create (from ISTALayerEnum)
            n_functions: Number of functions
            alpha: Alpha parameter
            lambd: Lambda parameter
            evaluation_func: Evaluation function to use
            logger: Logger instance
            **kwargs: Additional keyword arguments for the layer
            
        Returns:
            An instance of the requested ISTALayer implementation
            
        Note:
            The factory must have the requested layer type registered in _layer_map
        """
        return ISTALayerFactory._layer_map[kind](
            n_functions=n_functions,
            alpha=alpha,
            lambd=lambd,
            evaluation_func=evaluation_func,
            logger=logger,
            **kwargs
        )