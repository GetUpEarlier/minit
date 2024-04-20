from dataclasses import dataclass

from ..core.operator import Operator


@dataclass
class MatrixMultiply(Operator):
    ...


@dataclass
class BatchMatrixMultiply(Operator):
    ...


@dataclass
class TriangleUpper(Operator):
    pass


@dataclass
class TriangleLower(Operator):
    diagonal: int
