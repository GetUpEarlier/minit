import math
import os
from typing import List, Tuple

import numpy
import torch
from minit.collective.tensor import CollectiveTensor
from minit.core.torch import TorchTensor
from minit.cuda.tensor import CUDATensor
from minit.distributed.communicator import DistributedCommunicator
from minit.nccl.library import launch_server
from minit.nccl.tensor import NCCLTensor
from minit.distributed.group import initialize_world
import minit.functional as F
from minit.graph import dump_graphviz
from minit.module import Module
from minit.core.tensor import Tensor
from minit.module.checkpoint import load_from_safetensors_index
from minit.module.list import ModuleList
import nvtx

from minit.chat.server import CacheInputSession, CacheKeyValueSession, Server
from minit.chat.sample import Top1Sampler
from minit.chat.template import Template
from minit.chat.tokenizer import HuggingFaceTokenizer
from minit.chat.agent import Agent
from minit.remote.controller import Controller
from minit.remote.registry import register_constructor, register_function
from minit.trace.function import trace_function
from minit.trace.executor import TraceGraphExecutor
import minit.cuda.dispatch
import minit.trace.dispatch
import minit.collective.dispatch
import minit.nccl.dispatch

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
        self.q_proj = self.register_module("q_proj", Linear(hidden_size, nr_querys * head_size))
        self.k_proj = self.register_module("k_proj", Linear(hidden_size, nr_key_values // 2 * head_size))
        self.v_proj = self.register_module("v_proj", Linear(hidden_size, nr_key_values // 2 * head_size))
        self.o_proj = self.register_module("o_proj", Linear(nr_querys * head_size, hidden_size))

    @nvtx.annotate("attention")
    def forward(self, x: Tensor, offset: Tensor, precomp_freqs_cos: Tensor, precomp_freqs_sin: Tensor, k_cache: Tensor, v_cache: Tensor):
        nr_key_values = (self.nr_querys // self.nr_groups) * 2
        assert len(x.shape) == 3
        q = self.q_proj.forward(x).rearrange("bs(hd)->bshd", {
            "h": self.nr_querys,
            "d": self.head_size,
        })
        k = self.k_proj.forward(x).rearrange("bs(hd)->bs(hr)d", {
            "h": nr_key_values // 2,
            "d": self.head_size,
            "r": self.nr_groups,
        })
        v = self.v_proj.forward(x).rearrange("bs(hd)->bs(hr)d", {
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
        output = self.o_proj.forward(output)
        return output, k_cache, v_cache


class FeedForward(Module):
    def __init__(self, hidden_size: int, internal_size: int) -> None:
        super().__init__()
        self.gate_proj = self.register_module("gate_proj", Linear(hidden_size, internal_size))
        self.down_proj = self.register_module("down_proj", Linear(internal_size, hidden_size))
        self.up_proj = self.register_module("up_proj", Linear(hidden_size, internal_size))

    @nvtx.annotate("feed_forward")
    def forward(self, x: Tensor):
        output0 = silu(self.gate_proj.forward(x))
        output1 = self.up_proj.forward(x)
        output2 = self.down_proj.forward(output0 * output1)
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
    self_attn: Attention
    mlp: FeedForward
    input_layernorm: RMSNorm
    post_attention_layernorm: RMSNorm

    def __init__(self, hidden_size: int, nr_querys: int, nr_groups: int, head_size: int, internal_size: int) -> None:
        super().__init__()
        self.self_attn = self.register_module("self_attn", Attention(hidden_size, nr_querys, nr_groups, head_size))
        self.mlp = self.register_module("mlp", FeedForward(hidden_size, internal_size))
        self.input_layernorm = self.register_module("input_layernorm", RMSNorm(hidden_size))
        self.post_attention_layernorm = self.register_module("post_attention_layernorm", RMSNorm(hidden_size))

    @nvtx.annotate("layer")
    def forward(self, x: Tensor, offset: Tensor, precomp_freqs_cos: Tensor, precomp_freqs_sin: Tensor, k_cache: Tensor, v_cache: Tensor):
        attention_output, k_cache, v_cache = self.self_attn.forward(self.input_layernorm.forward(x), offset, precomp_freqs_cos, precomp_freqs_sin, k_cache, v_cache)
        attention_output = x + attention_output
        ffn_output = self.mlp.forward(self.post_attention_layernorm.forward(attention_output))
        return (attention_output + ffn_output), k_cache, v_cache

    def create_kv_cache(self) -> Tuple[Tensor, Tensor]:
        return (
            CUDATensor.allocate((1, 0, self.self_attn.nr_querys, self.self_attn.head_size), dtype=default_dtype),
            CUDATensor.allocate((1, 0, self.self_attn.nr_querys, self.self_attn.head_size), dtype=default_dtype)
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


class Llama3(Module):
    embed_tokens: Embedding
    layers: ModuleList[TransformerBlock]
    freqs_cos: Tensor
    freqs_sin: Tensor
    norm: RMSNorm

    def __init__(self, embedding_size: int, nr_layers: int, hidden_size: int, nr_querys: int, nr_groups: int, head_size: int, internal_size: int) -> None:
        super().__init__()
        self.layers = self.register_module("layers", ModuleList())
        for _ in range(nr_layers):
            self.layers.append(TransformerBlock(hidden_size, nr_querys, nr_groups, head_size, internal_size))
        self.embed_tokens = self.register_module("embed_tokens", Embedding(embedding_size, hidden_size))
        self.norm = self.register_module("norm", RMSNorm(hidden_size))
        self.freqs_cos, self.freqs_sin = self.precompute_freqs(head_size, 1024, 5e5)

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
        states = self.embed_tokens.forward(input_ids)
        freqs_cos = self.freqs_cos[offset:offset+s]
        freqs_sin = self.freqs_sin[offset:offset+s]
        all_states = []
        for i, block in enumerate(self.layers):
            all_states.append(states)
            block: TransformerBlock
            k_cache, v_cache = kv_caches[i]
            states, k_cache, v_cache = block.forward(states, offset, freqs_cos, freqs_sin, k_cache, v_cache)
            kv_caches[i] = k_cache, v_cache
        states = self.norm.forward(states)
        return states

    def create_kv_caches(self) -> List[Tuple[Tensor, Tensor]]:
        kv_caches = []
        for _i, block in enumerate(self.layers):
            kv_caches.append(block.create_kv_cache())
        return kv_caches
    

class Llama3LM(Module):
    model: Llama3
    lm_head: Embedding

    def __init__(self, embedding_size: int, nr_layers: int, hidden_size: int, nr_querys: int, nr_groups: int, head_size: int, internal_size: int) -> None:
        super().__init__()
        self.model = self.register_module("model", Llama3(embedding_size, nr_layers, hidden_size, nr_querys, nr_groups, head_size, internal_size))
        self.lm_head = self.register_module("lm_head", Embedding(embedding_size, hidden_size))

    def forward(self, input_ids: Tensor, offset: Tensor, kv_caches: List[Tuple[Tensor, Tensor]]):
        output = self.model.forward(input_ids, offset, kv_caches)
        output_probs = self.lm_head.backward(output)
        return output_probs

    def create_kv_caches(self) -> List[Tuple[Tensor, Tensor]]:
        return self.model.create_kv_caches()
    

class Llama3Server(Server[CacheInputSession]):
    model: Llama3LM
    model_forward = None

    def __init__(self, path: str) -> None:
        super().__init__()
        self.model = Llama3LM(
            embedding_size=128256,
            nr_layers=32,
            hidden_size=4096,
            nr_querys=32,
            nr_groups=4,
            head_size=128,
            internal_size=14336,
        )
        load_from_safetensors_index(self.model, path)

    def decode(self, session: CacheInputSession, input_ids: List[int]) -> Tensor:
        session.input_ids.extend(input_ids)
        input = CUDATensor.from_numpy(torch.tensor([session.input_ids], dtype=torch.int32))
        output_probs = self.model.forward(input, F.constant(0, "int32"), self.model.create_kv_caches())
        return output_probs[0][output_probs.shape[1]-1]

    def create_session(self) -> CacheInputSession:
        session = CacheInputSession()
        return session
    

class Llama3CollectiveServer(Server[CacheInputSession]):
    communicator: DistributedCommunicator
    model: Llama3LM

    def __init__(self, path: str, communicator: DistributedCommunicator) -> None:
        super().__init__()
        self.model = Llama3LM(
            embedding_size=128256,
            nr_layers=80,
            hidden_size=8192,
            nr_querys=64,
            nr_groups=8,
            head_size=128,
            internal_size=28672,
        )
        def epilogue(key: str, value: TorchTensor):
            value = CUDATensor.from_numpy(value.value)
            value = CollectiveTensor.from_broadcast(communicator, value)
            if "q_proj.weight" in key or "k_proj.weight" in key or "v_proj.weight" in key:
                value = value.to_split(0)
            elif "o_proj.weight" in key:
                value = value.to_split(1)
            elif "up_proj.weight" in key or "gate_proj.weight" in key:
                value = value.to_split(0)
            elif "down_proj.weight" in key:
                value = value.to_split(1)
            return value
        load_from_safetensors_index(self.model, path, epilogue)
        self.communicator = communicator

    def decode(self, session: CacheInputSession, input_ids: List[int]) -> Tensor:
        session.input_ids.extend(input_ids)
        input = CUDATensor.from_numpy(torch.tensor([session.input_ids], dtype=torch.int32))
        offset = F.constant(0, "int32")
        kv_caches = self.model.create_kv_caches()
        output_probs = self.model.forward(input, offset, kv_caches)
        return output_probs[0][output_probs.shape[1]-1]

    def create_session(self) -> CacheInputSession:
        session = CacheInputSession()
        return session
    

class Llama3TracedServer(Server[CacheInputSession]):
    model: Llama3LM
    model_forward = None

    def __init__(self, path: str) -> None:
        super().__init__()
        self.model = Llama3LM(
            embedding_size=128256,
            nr_layers=32,
            hidden_size=4096,
            nr_querys=32,
            nr_groups=4,
            head_size=128,
            internal_size=14336,
        )
        load_from_safetensors_index(self.model, path)

    def decode(self, session: CacheInputSession, input_ids: List[int]) -> Tensor:
        session.input_ids.extend(input_ids)
        input = CUDATensor.from_numpy(torch.tensor([session.input_ids], dtype=torch.int32))
        def flatten_kv_caches(kv_cache_list):
            kv_caches = []
            for k, v in kv_cache_list:
                kv_caches.append(k)
                kv_caches.append(v)
            return kv_caches

        def unflatten_kv_caches(kv_caches):
            kv_cache_list = []
            for i in range(len(kv_caches)//2):
                kv_cache_list.append((kv_caches[i*2], kv_caches[i*2+1]))
            return kv_cache_list

        if self.model_forward is None:
            def model_forward(input, offset, *kv_caches):
                kv_cache_list = unflatten_kv_caches(kv_caches)
                output_probs = self.model.forward(input, offset, kv_cache_list)
                kv_caches = flatten_kv_caches(kv_cache_list)
                return (output_probs, *kv_caches)
            model_forward = trace_function(model_forward, (input, F.constant(0, "int32"), *flatten_kv_caches(self.model.create_kv_caches())))
            self.model_forward = TraceGraphExecutor(model_forward)
            dump_graphviz(model_forward).save("graph.dot")
            # (output_probs, *kv_caches) = model_forward(input, F.constant(0, "int32"), *flatten_kv_caches(self.model.create_kv_caches()))
        (output_probs, *kv_caches) = self.model_forward.execute(input, F.constant(0, "int32"), *flatten_kv_caches(self.model.create_kv_caches()))
        # output_probs = self.model.forward(input, F.constant(0, "int32"), self.model.create_kv_caches())
        return output_probs[0][output_probs.shape[1]-1]

    def create_session(self) -> CacheInputSession:
        session = CacheInputSession()
        return session
    

class Llama3CachedServer(Server[CacheKeyValueSession]):
    model: Llama3LM

    def __init__(self, path: str) -> None:
        super().__init__()
        self.model = Llama3LM(
            embedding_size=128256,
            nr_layers=32,
            hidden_size=4096,
            nr_querys=32,
            nr_groups=4,
            head_size=128,
            internal_size=14336,
        )
        load_from_safetensors_index(self.model, path)

    def decode(self, session: CacheKeyValueSession, input_ids: List[int]) -> Tensor:
        input = CUDATensor.from_numpy(torch.tensor([input_ids], dtype=torch.int32))
        output_probs = self.model.forward(input, F.constant(session.offset, "int32"), session.kv_cache_list)
        session.offset += len(input_ids)
        return output_probs[0][output_probs.shape[1]-1]

    def create_session(self) -> CacheKeyValueSession:
        session = CacheKeyValueSession(self.model.create_kv_caches())
        return session


class Llama3Template(Template):
    def generate_header(self) -> str:
        return "<|begin_of_text|>"

    def generate_prompt(self, prompt: str) -> str:
        return f"<|start_header_id|>system<|end_header_id|>\n\n{prompt}<|eot_id|>"

    def generate(self, user: str) -> Tuple[str, str]:
        return f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", "<|eot_id|>"

    def eos(self) -> str:
        return "<|eot_id|>"
    

@register_constructor
def nccl_connect(id, rank, world_size) -> DistributedCommunicator:
    communicator = DistributedCommunicator(NCCLTensor.connect(id, rank, world_size))
    initialize_world(rank, world_size)
    return communicator


# @register_constructor
# def create_server(path, communicator) -> Llama3CollectiveServer:
#     return Llama3CollectiveServer(path, communicator)


@register_constructor
def create_session(agent: Agent, prompt: str):
    return agent.create_session(prompt)


@register_function
def execute_session(session: Agent.Session, user: str):
    return session.interact(user)


@register_constructor
def create_agent(communicator: DistributedCommunicator):
    path = "/root/autodl-tmp/modelscope/LLM-Research/Meta-Llama-3-70B-Instruct/"
    tokenizer_path = os.path.join(path, "tokenizer.json")
    server = Llama3CollectiveServer(os.path.join(path, "model.safetensors.index.json"), communicator)
    tokenizer = HuggingFaceTokenizer(tokenizer_path)
    sampler = Top1Sampler()
    template = Llama3Template()
    return Agent(server, tokenizer, template, sampler)


def main():
    controller = Controller("localhost", 0)
    print(f"serving at {controller.address}:{controller.port}")
    communicators = []
    agents = []
    sessions = []
    nr_ranks = 4
    controller.wait_for_connect(nr_ranks)
    nccl_server = launch_server()
    for i in range(nr_ranks):
        communicators.append(controller.invoke_function(i, nccl_connect, nccl_server, i, nr_ranks))
    for i in range(nr_ranks):
        agents.append(controller.invoke_function(i, create_agent, communicators[i].get()))
    for i in range(nr_ranks):
        sessions.append(controller.invoke_function(i, create_session, agents[i].get(), ""))
    request = "Hello!"
    for i, session in enumerate(sessions):
        response = controller.invoke_function(i, execute_session, session.get(), request)
    print(response.get())


if __name__ == '__main__':
    main()

