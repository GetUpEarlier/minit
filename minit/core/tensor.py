from typing import Tuple
from .array import Array

class Tensor(Array["Tensor"]):
    @property
    def shape(self) -> Tuple[int, ...]:
        ...

    @property
    def dtype(self) -> str:
        ...
