from typing import Tuple

class Tensor:
    @property
    def shape(self) -> Tuple[int, ...]:
        ...

    @property
    def dtype(self) -> str:
        ...
