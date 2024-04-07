from dataclasses import dataclass
from typing import List

from ..core.operator import Operator


@dataclass
class Fold(Operator):
    start: int
    stop: int

@dataclass
class Expand(Operator):
    axis: int

@dataclass
class AddAxis(Operator):
    axis: int

class RemoveAxis(Operator):
    axis: int

@dataclass
class Broadcast(Operator):
    axis: int

@dataclass
class Transpose(Operator):
    axis_a: int
    axis_b: int
