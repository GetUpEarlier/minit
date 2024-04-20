import functools
from types import FunctionType


def cached():
    def decorator(function: FunctionType):
        memo = {}
        @functools.wraps(function)
        def decorated(*args):
            try:
                return memo[args]
            except KeyError:
                result = function(*args)
                memo[args] = result
                return result
        return decorated
    return decorator
