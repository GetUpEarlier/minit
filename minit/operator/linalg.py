from dataclasses import dataclass

from ..core.operator import Operator


class MatrixMultiply(Operator):
    ...


class BatchMatrixMultiply(Operator):
    ...


@dataclass
class TriangleUpper(Operator):
    diagonal: int


@dataclass
class TriangleLower(Operator):
    diagonal: int
