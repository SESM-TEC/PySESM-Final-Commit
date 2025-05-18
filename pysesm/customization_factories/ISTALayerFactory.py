from pysesm.enums.ISTALayerEnum import ISTALayerEnum
from pysesm.models.BaseISTALayer import BaseISTALayer
from pysesm.models.ISTALayer import ISTALayer
from pysesm.models.FISTALayer import FISTALayer

class ISTALayerFactory:
    """
    Factory class for creating ISTALayer instances of specified types.
    
    Provides a centralized way to instantiate different variants of ISTA algorithms
    (classic ISTA, FISTA, etc.) with consistent parameter handling.
    """
    
    _layer_map = {
        ISTALayerEnum.CLASSIC: ISTALayer,
        ISTALayerEnum.FISTA: FISTALayer,
        # ISTALayerEnum.ADAPTIVE: AdaptiveISTALayer,
    }
    
    @staticmethod
    def create(kind: ISTALayerEnum, 
               n_functions: int, 
               alpha: float, 
               lambd: float,
               evaluation_func: callable, 
               logger,
               optimizer=None,
               device=None,
               parameter_hook=None,
               debug=False,
               **kwargs
        ) -> BaseISTALayer:
            """Crea una instancia del tipo de ISTALayer especificado."""
            specific_params = {}
            if kind == ISTALayerEnum.FISTA:
                # Parámetro opcional para reiniciar el momento cada N iteraciones
                restart_every = kwargs.pop('restart_every', 0)
                specific_params['restart_every'] = restart_every

            layer_params = {
                'n_functions': n_functions,
                'alpha': alpha,
                'lambd': lambd,
                'evaluation_func': evaluation_func,
                'logger': logger,
                'optimizer': optimizer,
                'device': device,
                'parameter_hook': parameter_hook,
                'debug': debug,
                **specific_params,
                **kwargs
            }
            return ISTALayerFactory._layer_map[kind](**layer_params)