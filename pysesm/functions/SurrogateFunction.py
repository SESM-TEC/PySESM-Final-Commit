from logging import Logger

from abc import ABC, abstractmethod
import torch
from typing import List, Union
from ..base_types import TensorBatch

class SurrogateFunction(ABC):
    """Abstract base class for defining surrogate functions within the
    SESM (Sparse-Encoded Surrogate Model) architecture.

    A surrogate function is designed to approximate a target function
    by evaluating multiple input points (tensors `x`) based on
    provided parameters.  For example, a linear function
    implementation might evaluate a set of points using a defined
    slope and intercept, producing outputs accordingly.

    This class serves as a base for implementing such surrogate
    functions, supporting their integration into the SESM
    architecture, where sparse encoding and linear combinations of
    dictionary functions are key components.  It establishes a
    contract for all surrogate functions, such as Gaussian,
    Polynomial, etc. It provides a polymorphic evaluation interface
    (`__call__`) that transparently handles different input data
    structures (dense tensors, nested_tensors for irregular batches, or
    lists of tensors).

    Inherited classes only need to implement the `evaluate` method,
    which contains the mathematical logic for a single 2D data tensor.

    Attributes:
        n_features (int):
            The size of each input tensor `x`, representing the number
            of features (or dimensions) the model works with.  For
            instance, in a dataset with `n_features=3`, each input
            point `x` would have three values.
        n_functions (int):
            The number of individual functions available in the
            internal dictionary.  These functions can be combined
            linearly as part of the SESM framework to approximate
            complex surrogate behaviors.
        logger (logging.Logger):
            A custom logger instance used to record runtime
            information during the execution of the surrogate
            function.  This enables detailed tracking of key events,
            debugging, and monitoring during model use.

    """

    @abstractmethod
    def __init__(self, n_features: int, n_functions: int, logger: Logger):
        """Function that initializes the approximate surrogate
        function with the given parameters

        Args:
            n_features (int):
                The size of each input tensor `x`, representing the
                number of features (or dimensions) the model works
                with.  For instance, in a dataset with `n_features=3`,
                each input point `x` would have three values.

            n_functions (int):        
                The number of individual functions available in the
                internal dictionary.  These functions can be combined
                linearly as part of the SESM framework to approximate
                complex surrogate behaviors.

            logger (logging.Logger):        
                A custom logger instance used to record runtime
                information during the execution of the surrogate
                function.  This enables detailed tracking of key
                events, debugging, and monitoring during model use.

        """
        self.n_features = n_features
        self.n_functions = n_functions
        self.logger = logger

    @staticmethod
    def _is_nested(X: torch.Tensor) -> bool:
        """
        Private helper to robustly check if a tensor is a NestedTensor.
        This encapsulates the check in a single place.
        """
        return getattr(X,"is_nested",False)

    @abstractmethod
    def initialize(self) -> torch.nn.Parameter:
        """Abstract method for initializing the parameters of the
        surrogate function.

        This method sets up and returns the parameters of the
        surrogate function that can be fitted during training.  The
        parameters are initialized based on the configuration defined
        during the instantiation of the surrogate function.

        Returns:
            torch.nn.Parameter:
                A vector (or tensor) of parameters wrapped as
                `torch.nn.Parameter`, which can be optimized during
                training.

        Notes:
            - Concrete implementations of this method should define
              specific attributes or parameters required for the
              surrogate function (e.g., weights, biases,
              coefficients).       
            - The initialized parameters are designed to integrate
              seamlessly with PyTorch's optimization framework.
        """
        pass
    
    # @torch.compile      # Disabled because it seems to generate slower code.
    def __call__(
        self,
        X: TensorBatch,
        *args, **kwargs
    ) -> TensorBatch:
        """Evaluates the surrogate function on a dataset.

        This is the main, polymorphic entry point. It can handle different
        input data structures by delegating the evaluation logic for a single
        tensor to the `evaluate` method.

        Args:
            X: The input data. It can be:
                - A standard `torch.Tensor` of shape (n_samples, n_features).
                - A `torch.nested.NestedTensor` for batches of irregular size.
                - A `list[torch.Tensor]` where each tensor is a batch.

            *args: Additional positional arguments to pass to the
                   `evaluate` method.

            **kwargs: Additional keyword arguments to pass to the
                      `evaluate` method.

        Returns:
            The evaluated output, maintaining the input data structure
            (Tensor, nested_tensor, or list of Tensors).
        """
        if isinstance(X, torch.Tensor) and not self._is_nested(X) and X.dim() == 2:
            # Base case: a single data tensor (n_samples, n_features).
            return self.evaluate(X, *args, **kwargs)

        elif self._is_nested(X):
            # NestedTensor case: unpack, process, and repack.
            # torch.compile will optimize this loop.

            list_of_tensors = X.unbind()
            results = [self.evaluate(tensor, *args, **kwargs) for tensor in list_of_tensors]
            return torch.nested.as_nested_tensor(results, layout=X.layout, device=X.device, dtype=results[0].dtype)

        elif isinstance(X, list):
            # List of tensors case: process each one.
            # torch.compile can also optimize this pattern.
            return [self.evaluate(tensor, *args, **kwargs) for tensor in X]

        else:
            # Handle edge cases or unsupported types.
            if self._is_nested(X) and X.dim() == 2:
                 # A 2D tensor can sometimes be flagged as nested; treat it as a regular tensor.
                 return self.evaluate(X, *args, **kwargs)
            
            raise TypeError(
                f"Unsupported input type for SurrogateFunction: {type(X)} "
                f"with dimensions {X.dim() if isinstance(X, torch.Tensor) else 'N/A'}"
            )

    @abstractmethod
    def evaluate(self, X: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Abstract method to evaluate the function on a single 2D data tensor.

        Child classes MUST implement this method. It defines the
        specific evaluation logic (e.g., Gaussian, Polynomial) for an
        input tensor of shape (n_samples, n_features) and must return
        a tensor of shape (n_samples, n_functions).

        Note that __call__ is the general method able to process
        several data batches, while this method processes a single
        batch.

        Args:
            X (torch.Tensor): The input tensor of shape (n_samples, n_features).
            *args: Additional positional arguments (e.g., Theta).
            **kwargs: Additional keyword arguments (e.g., mu_flag).

        Returns:
            torch.Tensor: The output tensor of shape (n_samples, n_functions).

        """
        pass
