from dataclasses import dataclass
from typing import List


@dataclass
class Fold:
    start: int
    stop: int

@dataclass
class Expand:
    axis: int

@dataclass
class AddAxis:
    axis: int

class RemoveAxis:
    axis: int

@dataclass
class Broadcast:
    axis: int

@dataclass
class Transpose:
    axis_a: int
    axis_b: int
