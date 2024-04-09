from dataclasses import dataclass
from ..core.operator import Operator


@dataclass
class Dequantize(Operator):
    axis: int
