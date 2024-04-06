from types import FunctionType
from typing import Tuple
from .operator import Operator
from .tensor import Tensor
import inspect

DISPATCH_TABLE = {}

def dispatch(operator: Operator, *args: Tensor) -> Tuple[Tensor, ...]:
    outputs = DISPATCH_TABLE[type(operator)](operator, *args)
    return outputs

def register_dispatch(function: FunctionType):
    signature = inspect.signature(function)
    op_type = signature.parameters["op"].annotation
    print(f"registering {op_type}")
    DISPATCH_TABLE[op_type] = function
