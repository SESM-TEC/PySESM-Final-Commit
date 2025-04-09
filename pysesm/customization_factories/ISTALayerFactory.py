from pysesm.enums.ISTALayerEnum import ISTALayerEnum
from pysesm.models.BaseISTALayer import BaseISTALayer
from pysesm.models.ISTALayer import ISTALayer

class ISTALayerFactory:
    """Factory para crear instancias de ISTALayer según el tipo especificado."""
    
    _layer_map = {
        ISTALayerEnum.CLASSIC: ISTALayer,
        # Registrar aquí nuevas implementaciones:
        # ISTALayerEnum.FISTA: FISTALayer,
        # ISTALayerEnum.ADAPTIVE: AdaptiveISTALayer,
    }
    
    @staticmethod
    def create(kind: ISTALayerEnum, n_functions: int, alpha: float, lambd: float,
               evaluation_func: callable, logger, **kwargs) -> BaseISTALayer:
        """Crea una instancia del tipo de ISTALayer especificado."""
        return ISTALayerFactory._layer_map[kind](
            n_functions=n_functions,
            alpha=alpha,
            lambd=lambd,
            evaluation_func=evaluation_func,
            logger=logger,
            **kwargs
        )