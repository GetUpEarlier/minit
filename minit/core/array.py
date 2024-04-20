from typing import Callable, Dict, Generic, List, Optional, Tuple, TypeVar


_T = TypeVar("_T")


def _reversed(binary_operator: Callable[[_T, _T], _T]):
    def reversed(x: _T, y: _T) -> _T:
        return binary_operator(y, x)
    return reversed


class Array(Generic[_T]):
    def fold(self, start: int, stop: int):
        from ..functional.shape import fold
        return fold(self, start, stop)

    def expand(self, axis: int, sizes: Tuple[_T, ...]):
        from ..functional.shape import expand
        return expand(self, axis, sizes)

    def add_axis(self, axis: int, size: Optional[_T] = None):
        from ..functional.shape import add_axis
        return add_axis(self, axis, size)

    def remove_axis(self, axis: int):
        from ..functional.shape import remove_axis
        return remove_axis(self, axis)

    def fold(self, start: int, stop: int):
        from ..functional.shape import fold
        return fold(self, start, stop)

    def broadcast(self, axis: int, size: _T):
        from ..functional.shape import broadcast
        return broadcast(self, axis, size)

    def transpose(self, axis_a: int, axis_b: int):
        from ..functional.shape import transpose
        return transpose(self, axis_a, axis_b)

    def repeat(self, axis: int, size: _T):
        from ..functional.shape import repeat
        return repeat(self, axis, size)

    def repeat_interleaved(self, axis: int, size: _T):
        from ..functional.shape import repeat_interleaved
        return repeat_interleaved(self, axis, size)

    def sum(self, axis: int):
        from ..functional.reduce import sum
        return sum(self, axis)

    def mean(self, axis: int):
        from ..functional.reduce import mean
        return mean(self, axis)

    def max(self, axis: int):
        from ..functional.reduce import max
        return max(self, axis)

    def slice(self, start: _T, stop: _T, axis: int):
        from ..functional.index import slice
        return slice(self, start, stop, axis)

    def slice_set(self, start: _T, stop: _T, axis: int, value: _T):
        from ..functional.index import slice_set
        return slice_set(self, start, stop, axis, value)

    def index(self, index: _T, axis: int):
        from ..functional.index import index as index_get
        return index_get(self, index, axis)

    def index_set(self, index: _T, axis: int, value: _T):
        from ..functional.index import index_set
        return index_set(self, index, axis, value)

    def split(self, axis: int, sizes: Tuple[_T, ...]):
        from ..functional.index import split
        return split(self, axis, sizes)

    def add(self, y: _T):
        from ..functional.arith import add
        return add(self, y)

    def subtract(self, y: _T):
        from ..functional.arith import subtract
        return subtract(self, y)

    def multiply(self, y: _T):
        from ..functional.arith import multiply
        return multiply(self, y)

    def divide(self, y: _T):
        from ..functional.arith import divide
        return divide(self, y)

    def power(self, y: _T):
        from ..functional.arith import power
        return power(self, y)

    def exponential(self):
        from ..functional.arith import exponential
        return exponential(self)

    def square(self):
        from ..functional.arith import square
        return square(self)

    def square_root(self):
        from ..functional.arith import square_root
        return square_root(self)

    def sine(self):
        from ..functional.arith import sine
        return sine(self)

    def cosine(self):
        from ..functional.arith import cosine
        return cosine(self)

    def reinterpret(self, target: str):
        from ..functional.shape import reinterpret
        return reinterpret(self, target)

    def cast(self, dtype: str):
        from ..functional.arith import cast
        return cast(self, dtype)

    def rearrange(self, equation: str, variables: Optional[Dict[str, _T]]=None):
        from ..functional.einops import rearrange
        return rearrange(equation, self, variables)

    __add__ = add
    __radd__ = _reversed(add)
    __sub__ = subtract
    __rsub__ = _reversed(subtract)
    __mul__ = multiply
    __rmul__ = _reversed(multiply)
    __truediv__ = divide
    __rtruediv__ = _reversed(divide)
    __pow__ = power
    __rpow__ = _reversed(power)

    def __getitem__(self, index) -> _T:
        assert not isinstance(index, tuple)
        if isinstance(index, slice):
            assert index.step is None
            if index.start is None:
                if index.stop is None:
                    return self
                else:
                    return self.slice(0, index.stop, 0)
            else:
                if index.stop is None:
                    return self.slice(index.start, self.shape[0], 0)
                else:
                    return self.slice(index.start, index.stop, 0)
        else:
            return self.slice(index, index+1, 0).remove_axis(0)
