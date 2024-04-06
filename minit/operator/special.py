from dataclasses import dataclass


class Sigmoid:
    ...


@dataclass
class RMSNorm:
    axis: int
    eps: float


@dataclass
class RoPE:
    ...


@dataclass
class Softmax:
    axis: int
