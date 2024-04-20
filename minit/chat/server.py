from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Generic, List, Tuple, TypeVar

from ..core.tensor import Tensor
from ..functional.arith import constant


_Session = TypeVar("_Session")


class Server(Generic[_Session]):
    @abstractmethod
    def decode(self, session: _Session, input_ids: List[int]) -> Tensor:
        ...

    @abstractmethod
    def create_session(self) -> _Session:
        ...


@dataclass
class CacheInputSession:
    input_ids: List[int] = field(default_factory=list)


@dataclass
class CacheKeyValueSession:
    kv_cache_list: List[Tuple[Tensor, Tensor]]
    offset: int

    def __init__(self, kv_cache_list: List[Tuple[Tensor, Tensor]]):
        self.kv_cache_list = kv_cache_list
        self.offset = 0
