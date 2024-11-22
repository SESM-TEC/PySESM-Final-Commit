from logging import Logger

from abc import ABC, abstractmethod
import torch


class SurrogateFunction(ABC):
    """
    Abstract base class for defining surrogate functions within the SESM (Sparse-Encoded Surrogate Model) architecture.

    A surrogate function is designed to approximate a target function by evaluating multiple input points (tensors `x`)
    based on provided parameters.
    For example, a linear function implementation might evaluate a set of points using
    a defined slope and intercept, producing outputs accordingly.

    This class serves as a base for implementing such surrogate functions, supporting their integration into the SESM
    architecture, where sparse encoding and linear combinations of dictionary functions are key components.

    Attributes:
        n_features (int):
            The size of each input tensor `x`, representing the number of features (or dimensions) the model
            works with.
            For instance, in a dataset with `n_features=3`, each input point `x` would have three values.

        n_functions (int):
            The number of individual functions available in the internal dictionary.
            These functions can be
            combined linearly as part of the SESM framework to approximate complex surrogate behaviors.

        logger (logging.Logger):
            A custom logger instance used to record runtime information during the execution of the surrogate
            function.
            This enables detailed tracking of key events, debugging, and monitoring during model use.
    """

    @abstractmethod
    def __init__(self, n_features: int, n_functions: int, logger: Logger):
        """
        Function that initializes the approximate surrogate function with the given parameters

        Args:
            n_features (int):
                The size of each input tensor `x`, representing the number of features (or dimensions) the model
                works with.
                For instance, in a dataset with `n_features=3`, each input point `x` would have three values.

            n_functions (int):
                The number of individual functions available in the internal dictionary.
                These functions can be
                combined linearly as part of the SESM framework to approximate complex surrogate behaviors.

            logger (logging.Logger):
                A custom logger instance used to record runtime information during the execution of the surrogate
                function.
                This enables detailed tracking of key events, debugging, and monitoring during model use.
        """
        self.n_features = n_features
        self.n_functions = n_functions
        self.logger = logger

    @abstractmethod
    def initialize(self) -> torch.nn.Parameter:
        """
        Abstract method for initializing the parameters of the surrogate function.

        This method sets up and returns the parameters of the surrogate function that can be
        fitted during training.
        The parameters are initialized based on the configuration
        defined during the instantiation of the surrogate function.

        Returns:
            torch.nn.Parameter:
                A vector (or tensor) of parameters wrapped as `torch.nn.Parameter`, which can be
                optimized during training.

        Notes:
            - Concrete implementations of this method should define specific attributes or
              parameters required for the surrogate function (e.g., weights, biases, coefficients).
            - The initialized parameters are designed to integrate seamlessly with PyTorch's
              optimization framework.
        """

    @abstractmethod
    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """
        Abstract method to evaluate the surrogate function.

        This method processes a batch of input samples and evaluates them across all the functions
        available in the internal dictionary, producing a tensor of size `(n_samples, n_functions)`.
        Each row corresponds to the evaluation of a single input sample across all the surrogate
        functions.

        Args:
            *args:
                Positional arguments required for evaluating the surrogate function.
                Typically, this
                includes the input tensor `X` of shape `(n_samples, n_features)`, where `n_samples`
                is the number of input points and `n_features` is the dimensionality of each point.
            **kwargs:
                Additional keyword arguments that provide flexibility for passing optional parameters
                or configurations needed during evaluation.

        Returns:
            torch.Tensor:
                A tensor of shape `(n_samples, n_functions)` where each element `(i, j)` represents
                the result of applying the `j`-th function to the `i`-th input sample.

        Notes:
            - Subclasses must implement this method to define the logic for evaluating each input
              sample across the surrogate functions, using the model's internal parameters.
        """
        pass
