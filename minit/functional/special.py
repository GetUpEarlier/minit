from ..operator.special import Sigmoid
from ..core.dispatch import dispatch
from ..core.tensor import Tensor


def sigmoid(x: Tensor):
    (z,) = dispatch(Sigmoid(), x)
    return z
