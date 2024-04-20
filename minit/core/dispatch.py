from types import FunctionType
from typing import Callable, List, Optional, Protocol, Sequence, Tuple, Type, overload
from .operator import Operator
from .tensor import Tensor
from .object import FunctionSignature, Object, extract_function_signature, match_function_args
import inspect
import nvtx

DISPATCH_CACHE = {}
DISPATCH_TABLE: List[Tuple[FunctionSignature, FunctionType, Optional["DispatchPredicate"], int]] = []

def lookup_implementation_from_types(tys: Tuple[Type, ...]):
    matching = []
    max_priority = None
    for signature, func, predicate, priority in DISPATCH_TABLE:
        match_result = match_function_args(signature, tys)
        if match_result is not None:
            predicate_result = predicate is None or predicate(*tys)
            if predicate_result:
                matching.append((match_result, signature, func, predicate, priority))
                if max_priority is None or priority > max_priority:
                    max_priority = priority
    assert len(matching) > 0, "no matching function"
    selected = []
    for match_result, signature, func, predicate, priority in matching:
        if priority == max_priority:
            selected.append((match_result, signature, func, predicate, priority))
    assert len(selected) == 1, f"more than one {len(matching)} function matches"
    _, _, cache, _, _ = selected[0]
    return cache

@overload
def dispatch(operator: Operator, *args: Tensor) -> Tuple[Tensor, ...]:
    ...

# @nvtx.annotate("dispatch")
def dispatch(*args: Object) -> Tuple[Tensor, ...]:
    arg_types = tuple([arg.type() for arg in args])
    try:
        func = DISPATCH_CACHE[arg_types]
    except KeyError:
        func = lookup_implementation_from_types(arg_types)
        DISPATCH_CACHE[arg_types] = func
    outputs = func(*args)
    return outputs

class DispatchPredicate(Protocol):
    def __call__(self, *args: Type) -> bool:
        ...

DEFAULT_PRIORITY = 0

def register_dispatch(*, predicate: Optional[DispatchPredicate] = None, priority: int = DEFAULT_PRIORITY):
    def decorator(function: FunctionType):
        signature = extract_function_signature(function)
        print(f"registering {signature}")
        DISPATCH_TABLE.append((signature, function, predicate, priority))
    return decorator
