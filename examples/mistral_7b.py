import math
import os
from typing import List, Tuple

import numpy
import torch
from minit.cuda.tensor import CUDATensor
import minit.functional as F
from minit.module import Module
from minit.core.tensor import Tensor
from minit.module.checkpoint import load_from_torch
from minit.module.list import ModuleList
import minit.cuda.dispatch
import nvtx
import sentencepiece

from agent import Server, Sampler, Tokenizer, Template, Agent, chat_cmdline

default_dtype = "float16"

def debug_tensor(name: str, x: Tensor):
    print(f"{name} mean: {x.numpy().astype(numpy.float32).mean()} var: {x.numpy().astype(numpy.float32).var()}")

def dump_tensor(name: str, x: Tensor):
    numpy.save(f"{name}_minit", x.numpy())

def silu(x: Tensor) -> Tensor:
    return x * F.sigmoid(x)

class Linear(Module):
    weight: Tensor

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = self.register_buffer("weight", (out_features, in_features), default_dtype)

    def forward(self, x: Tensor):
        return F.matrix_multiply(x, self.weight)


class Attention(Module):
    hidden_size: int
    nr_querys: int
    nr_groups: int
    head_size: int

    def __init__(self, hidden_size: int, nr_querys: int, nr_groups: int, head_size: int) -> None:
        super().__init__()
        nr_key_values = (nr_querys // nr_groups) * 2
        self.hidden_size = hidden_size
        self.nr_querys = nr_querys
        self.nr_groups = nr_groups
        self.head_size = head_size
        self.wq = self.register_module("wq", Linear(hidden_size, nr_querys * head_size))
        self.wk = self.register_module("wk", Linear(hidden_size, nr_key_values // 2 * head_size))
        self.wv = self.register_module("wv", Linear(hidden_size, nr_key_values // 2 * head_size))
        self.wo = self.register_module("wo", Linear(nr_querys * head_size, hidden_size))

    @nvtx.annotate("attention")
    def forward(self, x: Tensor, offset: Tensor, precomp_freqs_cos: Tensor, precomp_freqs_sin: Tensor, k_cache: Tensor, v_cache: Tensor):
        nr_key_values = (self.nr_querys // self.nr_groups) * 2
        assert len(x.shape) == 3
        q = self.wq.forward(x).rearrange("bs(hd)->bshd", {
            "h": self.nr_querys,
            "d": self.head_size,
        })
        k = self.wk.forward(x).rearrange("bs(hd)->bs(hr)d", {
            "h": nr_key_values // 2,
            "d": self.head_size,
            "r": self.nr_groups,
        })
        v = self.wv.forward(x).rearrange("bs(hd)->bs(hr)d", {
            "h": nr_key_values // 2,
            "d": self.head_size,
            "r": self.nr_groups,
        })
        seqlen = q.shape[1]
        q = F.rope(q, precomp_freqs_cos, precomp_freqs_sin)
        k = F.rope(k, precomp_freqs_cos, precomp_freqs_sin)
        k = F.tie([k_cache, k], axis=1)
        v = F.tie([v_cache, v], axis=1)
        k_cache = k
        v_cache = v
        qk = F.einsum("bsnd,btnd->bnst", q, k)
        qk = qk / math.sqrt(self.head_size) # b, nr_heads, s, s
        mask = F.triangle_upper(F.fill(-math.inf, (seqlen, seqlen), qk.dtype), diagonal=1)
        mask = F.tie([
            F.fill(0, (seqlen, offset), dtype=default_dtype),
            mask,
        ], axis=1)
        qk = qk + mask.rearrange("st->bnst", {
            "b": qk.shape[0],
            "n": qk.shape[1],
        })
        qk_softmax = F.softmax(qk, 3)
        output = F.einsum("bnst,btnd->bs(nd)", qk_softmax, v)
        output = self.wo.forward(output)
        return output, k_cache, v_cache


class FeedForward(Module):
    def __init__(self, hidden_size: int, internal_size: int) -> None:
        super().__init__()
        self.w1 = self.register_module("w1", Linear(hidden_size, internal_size))
        self.w2 = self.register_module("w2", Linear(internal_size, hidden_size))
        self.w3 = self.register_module("w3", Linear(hidden_size, internal_size))

    @nvtx.annotate("feed_forward")
    def forward(self, x: Tensor):
        output0 = silu(self.w1.forward(x))
        output1 = self.w3.forward(x)
        output2 = self.w2.forward(output0 * output1)
        return output2


class RMSNorm(Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.weight = self.register_buffer("weight", (size,), default_dtype)

    @nvtx.annotate("rms_norm")
    def forward(self, x: Tensor):
        axis = len(x.shape) - 1
        return F.rms_norm(x, self.weight, axis, 1e-5)


class TransformerBlock(Module):
    attention: Attention
    feed_forward: FeedForward
    attention_norm: RMSNorm
    ffn_norm: RMSNorm

    def __init__(self, hidden_size: int, nr_querys: int, nr_groups: int, head_size: int, internal_size: int) -> None:
        super().__init__()
        self.attention = self.register_module("attention", Attention(hidden_size, nr_querys, nr_groups, head_size))
        self.feed_forward = self.register_module("feed_forward", FeedForward(hidden_size, internal_size))
        self.attention_norm = self.register_module("attention_norm", RMSNorm(hidden_size))
        self.ffn_norm = self.register_module("ffn_norm", RMSNorm(hidden_size))

    @nvtx.annotate("layer")
    def forward(self, x: Tensor, offset: Tensor, precomp_freqs_cos: Tensor, precomp_freqs_sin: Tensor, k_cache: Tensor, v_cache: Tensor):
        attention_output, k_cache, v_cache = self.attention.forward(self.attention_norm.forward(x), offset, precomp_freqs_cos, precomp_freqs_sin, k_cache, v_cache)
        attention_output = x + attention_output
        ffn_output = self.feed_forward.forward(self.ffn_norm.forward(attention_output))
        return (attention_output + ffn_output), k_cache, v_cache

    def create_kv_cache(self) -> Tuple[Tensor, Tensor]:
        return (
            CUDATensor.allocate((1, 0, self.attention.nr_querys, self.attention.head_size), dtype=default_dtype),
            CUDATensor.allocate((1, 0, self.attention.nr_querys, self.attention.head_size), dtype=default_dtype)
        )


class Embedding(Module):
    def __init__(self, size: int, dim: int) -> None:
        super().__init__()
        self.weight = self.register_buffer("weight", (size, dim), default_dtype)

    # [B, S] . [W, H] -> [B, S, H]
    @nvtx.annotate("embedding.forward")
    def forward(self, x: Tensor) -> Tensor:
        variables = {}
        flatten_x = x.rearrange("bs->(bs)", variables)
        output = self.weight.index(flatten_x, 0)
        return output.rearrange("(bs)->bs", variables)

    # [B, S, H] . [W, H] -> [B, S, W]
    @nvtx.annotate("embedding.backward")
    def backward(self, x: Tensor) -> Tensor:
        return F.matrix_multiply(x, self.weight)


class LlaMa2(Module):
    tok_embeddings: Embedding
    layers: ModuleList[TransformerBlock]
    output: Embedding
    freqs_cos: Tensor
    freqs_sin: Tensor
    norm: RMSNorm

    def __init__(self, embedding_size: int, nr_layers: int, hidden_size: int, nr_querys: int, nr_groups: int, head_size: int, internal_size: int) -> None:
        super().__init__()
        self.layers = self.register_module("layers", ModuleList())
        for _ in range(nr_layers):
            self.layers.append(TransformerBlock(hidden_size, nr_querys, nr_groups, head_size, internal_size))
        self.tok_embeddings = self.register_module("tok_embeddings", Embedding(embedding_size, hidden_size))
        self.output = self.register_module("output", Embedding(embedding_size, hidden_size))
        self.norm = self.register_module("norm", RMSNorm(hidden_size))
        self.freqs_cos, self.freqs_sin = self.precompute_freqs(head_size, 1024, 1e6)

    def precompute_freqs(self, dim: int, size: int, theta: float) -> Tuple[Tensor, Tensor]:
        freqs = (1/theta) ** F.generate_sequence(0, dim//2, 2/dim, dtype="float32")
        t = F.generate_sequence(0, size, 1, dtype="float32")
        freqs = F.einsum("a,b->ab", t, freqs)
        freqs_cos = freqs.cosine().cast(default_dtype)
        freqs_sin = freqs.sine().cast(default_dtype)
        return freqs_cos, freqs_sin

    @nvtx.annotate("llama")
    def forward(self, input_ids: Tensor, offset: Tensor, kv_caches: List[Tuple[Tensor, Tensor]]):
        _b, s = input_ids.shape[:2]
        states = self.tok_embeddings.forward(input_ids)
        freqs_cos = self.freqs_cos[offset:offset.item()+s.item()]
        freqs_sin = self.freqs_sin[offset:offset.item()+s.item()]
        all_states = []
        for i, block in enumerate(self.layers):
            all_states.append(states)
            block: TransformerBlock
            k_cache, v_cache = kv_caches[i]
            states, k_cache, v_cache = block.forward(states, offset, freqs_cos, freqs_sin, k_cache, v_cache)
            kv_caches[i] = k_cache, v_cache
        states = self.norm.forward(states)
        output_probs = self.output.backward(states)
        return output_probs
    
    def create_kv_caches(self) -> List[Tuple[Tensor, Tensor]]:
        kv_caches = []
        for _i, block in enumerate(self.layers):
            kv_caches.append(block.create_kv_cache())
        return kv_caches
    

class Llama2Session:
    input_ids: List[int]

    def __init__(self) -> None:
        self.input_ids = []
    

class Llama2Server(Server[Llama2Session]):
    model: LlaMa2

    def __init__(self, path: str) -> None:
        super().__init__()
        self.model = LlaMa2(
            embedding_size=32000,
            nr_layers=32,
            hidden_size=4096,
            nr_querys=32,
            nr_groups=4,
            head_size=128,
            internal_size=14336,
        )
        load_from_torch(self.model, path)

    def decode(self, session: Llama2Session, input_ids: List[int]) -> Tensor:
        session.input_ids.extend(input_ids)
        input = CUDATensor.from_numpy(torch.tensor([session.input_ids], dtype=torch.int32))
        output_probs = self.model.forward(input, F.constant(0, "int32"), self.model.create_kv_caches())
        return output_probs[0][output_probs.shape[1]-1]

    def create_session(self) -> Llama2Session:
        session = Llama2Session()
        return session


class Llama2CachedSession:
    kv_caches: List[Tuple[Tensor, Tensor]]
    offset: int

    def __init__(self, kv_caches: List[Tuple[Tensor, Tensor]]) -> None:
        self.kv_caches = kv_caches
        self.offset = 0
    

class Llama2CachedServer(Server[Llama2CachedSession]):
    model: LlaMa2

    def __init__(self, path: str) -> None:
        super().__init__()
        self.model = LlaMa2(
            embedding_size=32000,
            nr_layers=32,
            hidden_size=4096,
            nr_querys=32,
            nr_groups=4,
            head_size=128,
            internal_size=14336,
        )
        load_from_torch(self.model, path)

    def decode(self, session: Llama2CachedSession, input_ids: List[int]) -> Tensor:
        input = CUDATensor.from_numpy(torch.tensor([input_ids], dtype=torch.int32))
        output_probs = self.model.forward(input, F.constant(session.offset, "int32"), session.kv_caches)
        session.offset += len(input_ids)
        return output_probs[0][output_probs.shape[1]-1]

    def create_session(self) -> Llama2CachedSession:
        session = Llama2CachedSession(self.model.create_kv_caches())
        return session


class Llama2Tokenizer(Tokenizer):
    def __init__(self, path: str) -> None:
        super().__init__()
        self.sentencepiece = sentencepiece.SentencePieceProcessor(path)

    def tokenize(self, text: str) -> List[int]:
        [input_ids,] = self.sentencepiece.tokenize([
            text
        ])
        return input_ids

    def detokenize(self, ids: List[int]) -> str:
        text = ""
        for id in ids:
            text += self.sentencepiece.id_to_piece(id)
        return text
    

class Llama2Template(Template):
    def generate_prompt(self, prompt: str) -> str:
        return f"<s>[INST] {prompt}"

    def eos(self) -> str:
        return "</s>"

    def first_chat(self, chat: str) -> str:
        return f"{chat} [/INST]"

    def next_chat(self, chat: str) -> str:
        return f"</s><s>[INST] {chat} [/INST]"


class Llama2Sampler(Sampler):
    def sample(self, probs: Tensor) -> int:
        output_id = probs.numpy().argmax(axis=-1).item()
        return output_id


def main():
    path = "/home/ubuntu/Git/minit/Downloads/Mistral-7B-v0.2-Instruct/"
    weights_path = os.path.join(path, "consolidated.00.pth")
    tokenizer_path = os.path.join(path, "tokenizer.model")
    server = Llama2CachedServer(weights_path)
    tokenizer = Llama2Tokenizer(tokenizer_path)
    sampler = Llama2Sampler()
    template = Llama2Template()
    agent = Agent(server, tokenizer, template, sampler)
    chat_cmdline(agent, "")


if __name__ == '__main__':
    main()

