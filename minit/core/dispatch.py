from types import FunctionType
from typing import Callable, List, Optional, Protocol, Tuple, Type
from .operator import Operator
from .tensor import Tensor
from .object import FunctionSignature, extract_function_signature, match_function_args
import inspect

DISPATCH_CACHE = {}
DISPATCH_TABLE: List[Tuple[FunctionSignature, FunctionType, Optional["DispatchPredicate"], int]] = []

def lookup_implementation_from_types(*tys: Type):
    cache = DISPATCH_CACHE.get(tys, None)
    if cache is not None:
        return cache
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
    DISPATCH_CACHE[tys] = cache
    return cache

def dispatch(operator: Operator, *args: Tensor) -> Tuple[Tensor, ...]:
    func = lookup_implementation_from_types(operator.type(), *(arg.type() for arg in args))
    outputs = func(operator, *args)
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
