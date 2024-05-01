from abc import abstractmethod
from typing import Tuple


class Template:
    @abstractmethod
    def generate_header(self) -> str:
        ...

    @abstractmethod
    def generate_prompt(self, prompt: str) -> str:
        ...

    @abstractmethod
    def generate(self, user: str) -> Tuple[str, str]:
        ...

    @abstractmethod
    def eos(self) -> str:
        ...
