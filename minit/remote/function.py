from dataclasses import dataclass


@dataclass(frozen=True)
class FunctionRef:
    location: int
    name: str
