from dataclasses import dataclass


@dataclass
class Slice:
    axis: int

class SliceSet:
    ...

@dataclass
class Index:
    axis: int

class IndexSet:
    ...

@dataclass
class Split:
    axis: int

@dataclass
class Tie:
    axis: int
