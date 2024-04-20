from dataclasses import dataclass
from enum import Enum
from typing import Union


@dataclass(frozen=True)
class CollectiveSpecSplit:
    axis: int

@dataclass(frozen=True)
class CollectiveSpecPartial:
    ...

@dataclass(frozen=True)
class CollectiveSpecBroadcast:
    ...

@dataclass(frozen=True)
class CollectiveSpecUnique:
    rank: int

CollectiveSpec = Union[
    CollectiveSpecSplit,
    CollectiveSpecPartial,
    CollectiveSpecBroadcast,
    CollectiveSpecUnique
]
