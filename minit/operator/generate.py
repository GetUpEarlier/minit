from dataclasses import dataclass
from numbers import Number

from ..core.operator import Operator


class GenerateInterval(Operator):
    pass


class GenerateSequence(Operator):
    pass


@dataclass
class Fill(Operator):
    value: Number
    dtype: str
