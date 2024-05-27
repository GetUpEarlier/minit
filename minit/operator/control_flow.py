from dataclasses import dataclass
from typing import Sequence

from ..graph import SubGraph
from ..core.operator import Operator
from ..core.tensor import Tensor


class Block:
    def __call__(self, *args: Tensor) -> Sequence[Tensor]:
        ...


@dataclass
class ForLoop(Operator):
    body: SubGraph


@dataclass
class WhileLoop(Operator):
    body: SubGraph


@dataclass
class IfBlock(Operator):
    true_body: SubGraph
    false_body: SubGraph
