from dataclasses import dataclass

from ..core.operator import Operator


@dataclass
class DistributedSend(Operator):
    target: int

@dataclass
class DistributedRecv(Operator):
    source: int

@dataclass
class DistributedBroadcast(Operator):
    source: int

@dataclass
class DistributedAllReduce(Operator):
    ...

@dataclass
class DistributedAllGather(Operator):
    axis: int

@dataclass
class DistributedReduceScatter(Operator):
    axis: int

@dataclass
class DistributedScatter(Operator):
    source: int
    axis: int

@dataclass
class DistributedAllToAll(Operator):
    gather_axis: int
    scatter_axis: int
