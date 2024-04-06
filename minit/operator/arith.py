from dataclasses import dataclass
from typing import Any


class Add:
    ...

class Subtract:
    ...

class Multiply:
    ...

class Divide:
    ...

class Power:
    ...

class Exponential:
    ...

class Cosine:
    ...

class Sine:
    ...

@dataclass
class Constant:
    value: Any
    dtype: str
