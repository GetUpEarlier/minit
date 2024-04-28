from dataclasses import dataclass
from typing import Union

from .value import Value


@dataclass(frozen=True)
class ObjectRef:
    location: int
    id: int
