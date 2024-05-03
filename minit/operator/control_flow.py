from dataclasses import dataclass

from ..graph import SubGraph


@dataclass
class ForLoop:
    body: SubGraph


class WhileLoop:
    ...


class If:
    ...
