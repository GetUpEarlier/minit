from typing import Optional, Tuple

import numpy

from ..core.meta import MetaTensor

from ..distributed.communicator import DistributedCommunicator
from .group import get_world
from ..core.tensor import Tensor
from .spec import CollectiveSpecBroadcast, CollectiveSpec, CollectiveSpecSplit, CollectiveSpecUnique, CollectiveSpecPartial


class CollectiveTensor(Tensor):
    __slots__ = [
        "_communicator",
        "_shape",
        "_local",
        "_spec",
    ]

    _communicator: DistributedCommunicator
    _shape: Tuple[Tensor, ...]
    _local: Optional[Tensor]
    _spec: CollectiveSpec

    def __init__(self, communicator: DistributedCommunicator, local: Optional[Tensor], spec: CollectiveSpec, shape: Tuple[Tensor, ...]) -> None:
        super().__init__()
        self._communicator = communicator
        self._local = local
        self._spec = spec
        self._shape = shape

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._local.dtype

    @property
    def spec(self):
        return self._spec

    @staticmethod
    def from_broadcast(communicator: DistributedCommunicator, local: Tensor):
        return CollectiveTensor(communicator, local, CollectiveSpecBroadcast(), local.shape)

    @staticmethod
    def from_split(communicator: DistributedCommunicator, local: Tensor, axis: int):
        shape = local.shape
        shape = shape[:axis] + shape[axis] * get_world().size + shape[axis+1:]
        return CollectiveTensor(communicator, local, CollectiveSpecSplit(axis), shape)

    @staticmethod
    def from_unique(communicator: DistributedCommunicator, local: Tensor, rank: int):
        return CollectiveTensor(communicator, local, CollectiveSpecUnique(rank), local.shape)

    @staticmethod
    def from_partial(communicator: DistributedCommunicator, local: Tensor):
        return CollectiveTensor(communicator, local, CollectiveSpecPartial(), local.shape)

    def to_broadcast(self) -> "CollectiveTensor":
        if isinstance(self.spec, CollectiveSpecBroadcast):
            return self
        elif isinstance(self.spec, CollectiveSpecSplit):
            return CollectiveTensor.from_broadcast(self._communicator, self._communicator.all_gather(self._local, self.spec.axis))
        elif isinstance(self.spec, CollectiveSpecPartial):
            return CollectiveTensor.from_broadcast(self._communicator, self._communicator.all_reduce(self._local))
        elif isinstance(self.spec, CollectiveSpecUnique):
            return CollectiveTensor.from_broadcast(self._communicator, self._communicator.broadcast(self._local, self.spec.rank))
        else:
            assert False

    def numpy(self) -> numpy.array:
        return self.to_broadcast()._local.numpy()

    def to_partial(self):
        if isinstance(self.spec, CollectiveSpecPartial):
            return self
        self = self.to_broadcast()
        return CollectiveTensor.from_partial(self._communicator, self._local / get_world().size)

    def to_split(self, axis: int):
        if isinstance(self.spec, CollectiveSpecSplit) and self.spec.axis == axis:
            return self
        self = self.to_broadcast()
        rank = get_world().rank
        size = self.shape[axis] / get_world().size
        return CollectiveTensor.from_split(self._communicator, self._local.slice(size*rank, size*(rank+1), axis), axis)

    def to_unique(self, rank: int):
        if isinstance(self.spec, CollectiveSpecUnique) and self.spec.rank == rank:
            return self
        self = self.to_broadcast()
        return CollectiveTensor.from_unique(self._communicator, self._local if get_world().rank == rank else MetaTensor(self._local.shape, self._local.dtype), rank)
