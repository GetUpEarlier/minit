from ..operator.special import RMSNorm, RoPE, Sigmoid, Softmax
from ..core.dispatch import dispatch
from ..core.tensor import Tensor


def sigmoid(x: Tensor):
    (z,) = dispatch(Sigmoid(), x)
    return z


def rms_norm(x: Tensor, weight: Tensor, axis: int, eps: float) -> Tensor:
    assert isinstance(eps, float)
    assert eps > 0
    assert eps < 0.5
    (z,) = dispatch(RMSNorm(axis=axis, eps=eps), x, weight)
    return z


def rope(x: Tensor, freqs_cos: Tensor, freqs_sin: Tensor) -> Tensor:
    (z,) = dispatch(RoPE(), x, freqs_cos, freqs_sin)
    return z


def softmax(x: Tensor, axis: int) -> Tensor:
    (z,) = dispatch(Softmax(axis=axis), x)
    return z
