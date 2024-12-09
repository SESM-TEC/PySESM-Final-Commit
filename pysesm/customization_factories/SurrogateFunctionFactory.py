from logging import Logger

from pysesm.functions import SurrogateFunction, GaussianFunction
from pysesm.enums import SurrogateFunctionEnum

function_map = {
    SurrogateFunctionEnum.GAUSSIAN: GaussianFunction,
}


class SurrogateFunctionFactory:

    @staticmethod
    def make(
        kind: SurrogateFunctionEnum,
        n_features: int,
        n_functions: int,
        seed: int,
        logger: Logger,
        **kwargs
    ) -> SurrogateFunction:
        return function_map[kind](
            n_features=n_features, n_functions=n_functions, seed=seed, logger=logger, **kwargs
        )
