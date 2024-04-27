from dataclasses import dataclass
import inspect
from types import FunctionType
from typing import Generic, Literal, Optional, Tuple, Type, Union, get_args, get_origin
from typing_extensions import Self


class Object:
    def type(self) -> Type[Self]:
        raise NotImplementedError()


def simplify_type(ty: Type) -> Tuple[Type, ...]:
    if get_origin(ty) == Union:
        return tuple(
            simplified_arg
            for arg in get_args(ty)
            for simplified_arg in simplify_type(arg)
        )
    else:
        return (ty,)


@dataclass
class FunctionSignature:
    args: Tuple[Tuple[Type, ...], ...]
    vaargs: Optional[Type]


def extract_function_signature(fn: FunctionType) -> FunctionSignature:
    signature = inspect.signature(fn)
    args = []
    vaargs = None
    for param in signature.parameters.values():
        assert param.kind in [param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD, param.VAR_POSITIONAL]
        simplified = simplify_type(param.annotation)
        if param.kind == param.VAR_POSITIONAL:
            vaargs = simplified
        else:
            args.append(simplified)
    return FunctionSignature(args, vaargs)


def is_literal(ty: Type):
    return get_origin_or_self(ty) == Literal


def get_origin_or_self(ty: Type):
    return get_origin(ty) or ty


def is_union(ty: Type):
    return get_origin_or_self(ty) == Union


# mro without dedup
def generic_mro(ty: Type):
    if not hasattr(ty, "__origin__"):
        return inspect.getmro(ty)
    orig_bases = ty.__origin__.__orig_bases__
    generic = None
    for orig_base in orig_bases:
        if get_origin_or_self(orig_base) == Generic:
            generic = orig_base
    assert generic is not None, f"generic not found for {ty}"
    assert len(get_args(ty)) == len(get_args(generic))
    args_mapping = { generic_arg: arg for arg, generic_arg in zip(get_args(ty), get_args(generic), strict=True) }
    if len(orig_bases) == 1:
        return ty
    def substitude(ty: Type):
        if not hasattr(ty, "__origin__"):
            return ty
        args = tuple(map(lambda t: args_mapping.get(t, default=t), get_args(ty)))
        return get_origin_or_self(ty)[args]
    results = []
    for orig_base in orig_bases:
        if orig_base is generic:
            continue
        for orig_base_mro in generic_mro(substitude(orig_base)):
            results.append(orig_base_mro)
    return tuple(results)


def match_pattern(pattern: Type, arg: Type):
    # literal
    if is_literal(pattern):
        if is_literal(arg) and pattern == arg:
            return pattern
        return None
    # object
    if is_union(pattern):
        for pattern_arg in get_args(pattern):
            result = match_pattern(pattern_arg, arg)
            if result is not None:
                return result
        return None
    # arg literal
    origin_pattern = get_origin_or_self(pattern)
    if is_literal(arg):
        if isinstance(get_args(arg)[0], origin_pattern):
            return pattern
        return None
    origin_arg = get_origin_or_self(arg)
    if not issubclass(origin_arg, origin_pattern):
        return None
    if origin_arg != origin_pattern:
        for base in generic_mro(arg):
            if get_origin_or_self(base) == origin_pattern:
                arg = base
                origin_arg = get_origin_or_self(base)
        assert origin_arg == origin_pattern
    if get_args(pattern) == ():
        return pattern
    if get_args(arg) == ():
        return None
    for arg_arg, pattern_arg in zip(get_args(arg), get_args(pattern), strict=True):
        match_result = match_pattern(pattern_arg, arg_arg)
        if match_result is None:
            return None
    return pattern


def match_patterns(patterns: Tuple[Type, ...], arg: Type) -> Optional[Type]:
    for pattern in patterns:
        match_result = match_pattern(pattern, arg)
        if match_result is not None:
            return match_result
    return None


def match_function_args(signature: FunctionSignature, args: Tuple[Type, ...]):
    result = []
    if len(args) < len(signature.args):
        return None
    if len(args) > len(signature.args):
        if signature.vaargs is None:
            return None
    nr_args = len(signature.args)
    for i in range(nr_args):
        match_result = match_patterns(signature.args[i], args[i])
        if match_result is None:
            return None
        result.append(match_result)
    for arg in args[nr_args:]:
        match_result = match_patterns(signature.vaargs, arg)
        if match_result is None:
            return None
        result.append(match_result)
    return tuple(result)
