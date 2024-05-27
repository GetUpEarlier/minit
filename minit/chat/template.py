from abc import abstractmethod
from typing import Tuple


class Template:
    @abstractmethod
    def generate_header(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def generate_prompt(self, prompt: str) -> str:
        raise NotImplementedError()

    @abstractmethod
    def generate(self, user: str) -> Tuple[str, str]:
        raise NotImplementedError()

    @abstractmethod
    def eos(self) -> str:
        raise NotImplementedError()
