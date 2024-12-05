from logging import Logger

from pysesm.functions.SurrogateFunction import SurrogateFunction
from pysesm.functions.GaussianFunction import GaussianFunction
from pysesm.enums import SurrogateFunctionEnum

function_map = {
    "GaussianFunction": GaussianFunction,
}


class SurrogateFunctionFactory:

    @staticmethod
    def make(kind: SurrogateFunctionEnum, n_features: int, n_functions: int, logger: Logger, **kwargs) -> SurrogateFunction:
        return function_map[kind](n_features=n_features, n_functions=n_functions, logger=logger, **kwargs)
