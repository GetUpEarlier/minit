from typing import Tuple

from .object import Object
from .array import Array

class Tensor(Array["Tensor"], Object):
    @property
    def shape(self) -> Tuple["Tensor", ...]:
        raise NotImplementedError()

    @property
    def dtype(self) -> str:
        raise NotImplementedError()
    
    @property
    def device(self) -> str:
        raise NotImplementedError()

    def type(self):
        return type(self)
