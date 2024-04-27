from dataclasses import dataclass
from enum import Enum
from typing import Generic, Literal, TypeVar, Union

from ..core.object import Object


_Axis = TypeVar("_Axis")
_Rank = TypeVar("_Rank")


@dataclass(frozen=True)
class CollectiveSpecSplit(Object, Generic[_Axis]):
    axis: int

    def type(self):
        return CollectiveSpecSplit[Literal[self.axis]]

@dataclass(frozen=True)
class CollectiveSpecPartial(Object):
    def type(self):
        return CollectiveSpecPartial

@dataclass(frozen=True)
class CollectiveSpecBroadcast(Object):
    def type(self):
        return CollectiveSpecBroadcast

@dataclass(frozen=True)
class CollectiveSpecUnique(Object, Generic[_Rank]):
    rank: int

    def type(self):
        return CollectiveSpecUnique[Literal[self.rank]]

CollectiveSpec = Union[
    CollectiveSpecSplit[int],
    CollectiveSpecPartial,
    CollectiveSpecBroadcast,
    CollectiveSpecUnique[int],
]
