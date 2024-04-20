from abc import abstractmethod


class Template:
    @abstractmethod
    def generate_prompt(self, prompt: str) -> str:
        ...

    @abstractmethod
    def eos(self) -> str:
        ...

    @abstractmethod
    def first_chat(self, chat: str) -> str:
        ...

    @abstractmethod
    def next_chat(self, chat: str) -> str:
        ...
