from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class DistributedGroup:
    size: int
    rank: int


_WORLD = None


def get_world() -> DistributedGroup:
    return _WORLD


def initialize_world(rank: int, size: int):
    global _WORLD
    assert _WORLD is None
    _WORLD = DistributedGroup(size, rank)
