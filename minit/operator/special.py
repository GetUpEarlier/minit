from dataclasses import dataclass

from ..core.operator import Operator


class Sigmoid(Operator):
    ...


@dataclass
class RMSNorm(Operator):
    axis: int
    eps: float


@dataclass
class RoPE(Operator):
    ...


@dataclass
class Softmax(Operator):
    axis: int
