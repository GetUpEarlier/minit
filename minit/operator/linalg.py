from dataclasses import dataclass


class MatrixMultiply:
    ...


class BatchMatrixMultiply:
    ...


@dataclass
class TriangleUpper:
    diagonal: int


@dataclass
class TriangleLower:
    diagonal: int
