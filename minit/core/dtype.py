from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class DataTypeInfo:
    name: str
    python_type: Optional[type]
    size_in_bytes: int


DATA_TYPES: Dict[str, DataTypeInfo] = {}


def register_dtype(name: str, python_type: Optional[type], size_in_bytes: int):
    assert name not in DATA_TYPES
    DATA_TYPES[name] = DataTypeInfo(
        name, python_type, size_in_bytes
    )


def dtype_info(name: str) -> DataTypeInfo:
    return DATA_TYPES[name]


register_dtype("float64", float, 8)
register_dtype("float32", float, 4)
register_dtype("float16", float, 2)
register_dtype("bfloat16", float, 2)

register_dtype("int64", int, 8)
register_dtype("int32", int, 4)
register_dtype("int16", int, 2)
register_dtype("int8", int, 1)

register_dtype("uint64", int, 8)
register_dtype("uint32", int, 4)
register_dtype("uint16", int, 2)
register_dtype("uint8", int, 1)
