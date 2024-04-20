from abc import abstractmethod
from typing import Generator, Generic, List, Optional, TypeVar

from minit.core.tensor import Tensor


_Session = TypeVar("_Session")


class Server(Generic[_Session]):
    @abstractmethod
    def decode(self, session: _Session, input_ids: List[int]) -> Tensor:
        ...

    @abstractmethod
    def create_session(self) -> _Session:
        ...


class Tokenizer:
    @abstractmethod
    def tokenize(self, text: str) -> List[int]:
        ...

    @abstractmethod
    def detokenize(self, ids: List[int]) -> str:
        ...


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


class Sampler:
    @abstractmethod
    def sample(self, probs: Tensor) -> int:
        ...


class Agent:
    server: Server
    tokenizer: Tokenizer
    template: Template
    sampler: Sampler

    def __init__(self, server: Server, tokenizer: Tokenizer, template: Template, sampler: Sampler) -> None:
        self.server = server
        self.tokenizer = tokenizer
        self.template = template
        self.sampler = sampler

    def chat(self, prompt: str) -> Generator[Optional[str], str, None]:
        session = self.server.create_session()
        text = self.template.generate_prompt(prompt)
        question = yield None
        text += self.template.first_chat(question)
        end_chat = False
        while True:
            output_id = None
            response = ""
            while True:
                if output_id is not None:
                    input_ids = [output_id]
                else:
                    input_ids = self.tokenizer.tokenize(text)
                output_probs = self.server.decode(session, input_ids)
                output_id = self.sampler.sample(output_probs)
                text = self.tokenizer.detokenize([output_id])
                response += text
                if text == self.template.eos():
                    end_chat = True
                if end_chat:
                    end_chat = False
                    break
                pong = yield text
                assert pong is None
            question = yield None
            assert question is not None
            text = self.template.next_chat(question)


def chat_cmdline(agent: Agent, prompt: str):
    chat = agent.chat(prompt)
    assert chat.send(None) is None
    while True:
        print("User:", end="\t", flush=True)
        request = input()
        print("Assistent:", end="\t", flush=True)
        response = chat.send(request)
        while response is not None:
            print(response, end="", flush=True)
            response = chat.send(None)
        print()
