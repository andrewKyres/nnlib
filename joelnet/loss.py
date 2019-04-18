"""
A loss function measures how good our predictions are,
we can use this to adjust the parameters of our network
"""
import numpy as np

from joelnet.tensor import Tensor

class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError


class TSE(Loss):
    """
    Total Squared Error
    """
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.sum(np.power(predicted - actual, 2))

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2 * (predicted - actual)

class MSE(Loss):
    """
    Mean Squared Error
    """
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.mean(np.power(predicted - actual, 2))

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        """
        The derivative of the loss fn
        """
        return 2 * (predicted - actual)
