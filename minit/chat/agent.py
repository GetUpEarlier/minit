from typing import Callable, List, Optional

from ..core.tensor import Tensor
from .server import Server
from .tokenizer import Tokenizer
from .template import Template
from .sample import Sampler


class Agent:
    server: Server
    tokenizer: Tokenizer
    template: Template
    sampler: Sampler

    class Session:
        decoder: Callable[[List[int]], Tensor]
        tokenizer: Tokenizer
        template: Template
        sampler: Sampler

        def __init__(self, server: Server, tokenizer: Tokenizer, template: Template, sampler: Sampler, prompt: Optional[str]) -> None:
            session = server.create_session()
            def decoder(input_ids: List[int]):
                return server.decode(session, input_ids)
            self.decoder = decoder
            header = template.generate_header()
            if prompt is not None:
                header += template.generate_prompt(prompt)
            if len(header) > 0:
                self.decoder(tokenizer.tokenize(header))
            self.template = template
            self.sampler = sampler
            self.tokenizer = tokenizer

        def interact(self, user: str) -> str:
            prefix, postfix = self.template.generate(user)
            assistant = ""
            eos = self.template.eos()
            input_ids = self.tokenizer.tokenize(prefix)
            while True:
                output_probs = self.decoder(input_ids)
                output_id = self.sampler.sample(output_probs)
                output = self.tokenizer.detokenize([output_id])
                if output == eos:
                    break
                assistant += output
                input_ids = [output_id]
            self.decoder(self.tokenizer.tokenize(postfix))
            return assistant

    def __init__(self, server: Server, tokenizer: Tokenizer, template: Template, sampler: Sampler) -> None:
        self.server = server
        self.tokenizer = tokenizer
        self.template = template
        self.sampler = sampler

    def create_session(self, prompt: str) -> Session:
        return Agent.Session(self.server, self.tokenizer, self.template, self.sampler, prompt)
