from abc import abstractmethod
from ..core.tensor import Tensor


class Sampler:
    @abstractmethod
    def sample(self, probs: Tensor) -> int:
        ...


class Top1Sampler(Sampler):
    def sample(self, probs: Tensor) -> int:
        output_id = probs.numpy().argmax(axis=-1).item()
        return output_id
