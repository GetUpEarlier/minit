from dataclasses import dataclass
from numbers import Number


class GenerateInterval:
    pass


class GenerateSequence:
    pass


@dataclass
class Fill:
    value: Number
    dtype: str
