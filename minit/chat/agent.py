from typing import Generator, Optional

from .server import Server
from .tokenizer import Tokenizer
from .template import Template
from .sample import Sampler


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
