import logging

import torch


# TODO: Understanding how it works maybe this class could be thought as a Provider rather than a Function abstraction.
# TODO: As it doesnt hold any function values but more like configuration attributes
# TODO: Renamed: Function ->  Approximate Surrogate Function
class ApproximateSurrogateFunction:
    """
    Class that defines the approximate surrogate function interface that should be used along with the SESM architecture.

    Attributes:
        n_features (int): The number of input features or dimensions.
        n_functions (int): The number of functions or basis functions.
        logger (logging.Logger): Logger instance to be used
    """

    def __init__(self, n_features: int, n_functions: int, logger: logging.Logger):
        """
        Function that initializes the approximate surrogate function with the given parameters

        Args:
            n_features (int): The number of input features or dimensions.
            n_functions (int): The number of functions or basis functions.
            logger (logging.Logger): Logger instance to be used
        """
        self.n_features = n_features
        self.n_functions = n_functions
        self.logger = logger

    def initialize(self):
        pass

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        pass
