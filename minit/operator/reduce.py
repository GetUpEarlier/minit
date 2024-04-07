from dataclasses import dataclass

from ..core.operator import Operator


@dataclass
class Sum(Operator):
    axis: int


@dataclass
class Max(Operator):
    axis: int
