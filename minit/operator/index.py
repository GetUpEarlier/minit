from dataclasses import dataclass

from ..core.operator import Operator


@dataclass
class Slice(Operator):
    axis: int

class SliceSet(Operator):
    ...

@dataclass
class Index(Operator):
    axis: int

class IndexSet(Operator):
    ...

@dataclass
class Split(Operator):
    axis: int

@dataclass
class Tie(Operator):
    axis: int
