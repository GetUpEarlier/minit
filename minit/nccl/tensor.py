from ..core.tensor import Tensor
from .library import connect_server


class NCCLTensor(Tensor):
    comm: int

    def __init__(self, comm: int) -> None:
        super().__init__()
        self.comm = comm

    @property
    def shape(self):
        return ()

    @property
    def dtype(self):
        return "int64"

    @staticmethod
    def connect(unique_id: bytes, rank: int, size: int) -> "NCCLTensor":
        comm = connect_server(size, rank, unique_id)
        return NCCLTensor(comm)
