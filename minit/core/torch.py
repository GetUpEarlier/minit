from .tensor import Tensor
import torch


class TorchTensor(Tensor):
    def __init__(self, value: torch.Tensor) -> None:
        super().__init__()
        self._value = value

    @property
    def value(self):
        return self._value

    @property
    def device(self):
        return "torch"
