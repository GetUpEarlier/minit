import math
from typing import Dict, Tuple

import numpy
from minit.cuda.tensor import CUDATensor
import minit.functional as F
from minit.functional.generate import fill, generate_sequence
from minit.functional.shape import add_axis, broadcast, fold, expand, remove_axis, repeat, repeat_interleaved, transpose
from minit.functional.index import split, tie
from minit.functional.special import rms_norm, rope, sigmoid, softmax
from minit.module import Module
from minit.core.tensor import Tensor
from minit.functional.arith import add, constant, cosine, divide, exponential, power, multiply, sine, square, square_root, subtract
from minit.functional.linalg import batch_matrix_multiply, matrix_multiply, triangle_lower, triangle_upper
from minit.functional.index import index, slice
from minit.functional.reduce import mean, sum, max
from minit.module.checkpoint import load_from_safetensors, load_from_torch
from minit.module.list import ModuleList
import minit.cuda.dispatch
import nvtx

default_dtype = "float16"

def debug_tensor(name: str, x: Tensor):
    print(f"{name} mean: {x.numpy().astype(numpy.float32).mean()} var: {x.numpy().astype(numpy.float32).var()}")

def dump_tensor(name: str, x: Tensor):
    numpy.save(f"{name}_minit", x.numpy())

def silu(x: Tensor) -> Tensor:
    return multiply(x, sigmoid(x))


class Linear(Module):
    weight: Tensor

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = self.register_buffer("weight", (out_features, in_features), default_dtype)

    def forward(self, x: Tensor):
        return matrix_multiply(x, self.weight)


class LinearWithBias(Module):
    weight: Tensor

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = self.register_buffer("weight", (out_features, in_features), default_dtype)
        self.bias = self.register_buffer("bias", (out_features,), default_dtype)

    def forward(self, x: Tensor):
        output = matrix_multiply(x, self.weight)
        bias = self.bias
        while len(bias.shape) < len(x.shape):
            bias = bias.add_axis(0, x.shape[-1-len(bias.shape)])
        output += bias
        return output


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
        self.q_proj = self.register_module("q_proj", LinearWithBias(hidden_size, nr_querys * head_size))
        self.k_proj = self.register_module("k_proj", LinearWithBias(hidden_size, nr_key_values // 2 * head_size))
        self.v_proj = self.register_module("v_proj", LinearWithBias(hidden_size, nr_key_values // 2 * head_size))
        self.o_proj = self.register_module("o_proj", Linear(nr_querys * head_size, hidden_size))

    @nvtx.annotate("attention")
    def forward(self, x: Tensor, precomp_freqs_cos: Tensor, precomp_freqs_sin: Tensor):
        nr_key_values = (self.nr_querys // self.nr_groups) * 2
        assert len(x.shape) == 3
        q = self.q_proj(x)
        q = q.rearrange("bs(nd)->bsnd", dict(
            n=self.nr_querys,
            d=self.head_size,
        ))
        k = self.k_proj(x)
        k = k.rearrange("bs(nd)->bsnd", dict(
            n=nr_key_values // 2,
            d=self.head_size,
        ))
        v = self.v_proj(x)
        v = v.rearrange("bs(nd)->bsnd", dict(
            n=nr_key_values // 2,
            d=self.head_size,
        ))
        k = k.repeat_interleaved(2, self.nr_groups)
        v = v.repeat_interleaved(2, self.nr_groups)
        q = rope(q, precomp_freqs_cos, precomp_freqs_sin)
        k = rope(k, precomp_freqs_cos, precomp_freqs_sin)
        qk = F.einsum("bsnd,btnd->bnst", q, k)
        qk = qk / math.sqrt(self.head_size)
        qk = qk + triangle_upper(fill(-math.inf, qk.shape, qk.dtype), diagonal=1)
        qk_softmax = softmax(qk, 3)
        output = F.einsum("bnst,btnd->bs(nd)", qk_softmax, v)
        output = self.o_proj(output)
        return output


class FeedForward(Module):
    def __init__(self, hidden_size: int, internal_size: int) -> None:
        super().__init__()
        self.gate_proj = self.register_module("gate_proj", Linear(hidden_size, internal_size))
        self.down_proj = self.register_module("down_proj", Linear(internal_size, hidden_size))
        self.up_proj = self.register_module("up_proj", Linear(hidden_size, internal_size))

    @nvtx.annotate("feed_forward")
    def forward(self, x: Tensor):
        output0 = silu(self.gate_proj(x))
        output1 = self.up_proj(x)
        output2 = self.down_proj(output0 * output1)
        return output2


class RMSNorm(Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.weight = self.register_buffer("weight", (size,), default_dtype)

    @nvtx.annotate("rms_norm")
    def forward(self, x: Tensor):
        axis = len(x.shape) - 1
        return rms_norm(x, self.weight, axis, 1e-6)


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
    def forward(self, x: Tensor, precomp_freqs_cos: Tensor, precomp_freqs_sin: Tensor):
        attention_output = x + self.self_attn.forward(self.input_layernorm.forward(x), precomp_freqs_cos, precomp_freqs_sin)
        ffn_output = self.mlp.forward(self.post_attention_layernorm.forward(attention_output))
        return attention_output + ffn_output


class Embedding(Module):
    def __init__(self, size: int, dim: int) -> None:
        super().__init__()
        self.weight = self.register_buffer("weight", (size, dim), default_dtype)

    # [B, S] . [W, H] -> [B, S, H]
    @nvtx.annotate("embedding.forward")
    def forward(self, x: Tensor) -> Tensor:
        flatten_x = x.fold(0, len(x.shape))
        output = self.weight.index(flatten_x, 0)
        return output.expand(0, x.shape)

    # [B, S, H] . [W, H] -> [B, S, W]
    @nvtx.annotate("embedding.backward")
    def backward(self, x: Tensor) -> Tensor:
        return matrix_multiply(x, self.weight)


class LlaMa2Model(Module):
    embed_tokens: Embedding
    layers: ModuleList[TransformerBlock]
    freqs_cos: Tensor
    freqs_sin: Tensor
    norm: RMSNorm

    def __init__(self,
                embedding_size: int,
                nr_layers: int,
                hidden_size: int,
                nr_querys: int,
                nr_groups: int,
                head_size: int,
                internal_size: int
            ) -> None:
        super().__init__()
        self.layers = self.register_module("layers", ModuleList())
        for _ in range(nr_layers):
            self.layers.append(TransformerBlock(hidden_size, nr_querys, nr_groups, head_size, internal_size))
        self.embed_tokens = self.register_module("embed_tokens", Embedding(embedding_size, hidden_size))
        self.norm = self.register_module("norm", RMSNorm(hidden_size))
        self.freqs_cos, self.freqs_sin = self.precompute_freqs(head_size, 1024, 1e6)

    def precompute_freqs(self, dim: int, size: int, theta: float) -> Tuple[Tensor, Tensor]:
        freqs = fill(1.0, (dim//2,), "float32") / fill(theta, (dim//2,), "float32") ** generate_sequence(0, dim//2, 2/dim, dtype="float32")
        freqs = freqs.add_axis(1)
        t = generate_sequence(0, size, 1, dtype="float32")
        t = t.add_axis(1)
        freqs = matrix_multiply(t, freqs)
        freqs_cos = freqs.cosine()
        freqs_sin = freqs.sine()
        freqs_cos = CUDATensor.from_numpy(freqs_cos.numpy().astype(getattr(numpy, default_dtype)))
        freqs_sin = CUDATensor.from_numpy(freqs_sin.numpy().astype(getattr(numpy, default_dtype)))
        return freqs_cos, freqs_sin

    @nvtx.annotate("llama")
    def forward(self, input_ids: Tensor):
        _b, s = input_ids.shape[:2]
        states = self.embed_tokens.forward(input_ids)
        freqs_cos = self.freqs_cos.slice(0, s, 0)
        freqs_sin = self.freqs_sin.slice(0, s, 0)
        all_states = []
        for _i, block in enumerate(self.layers):
            all_states.append(states)
            block: TransformerBlock
            states = block.forward(states, freqs_cos, freqs_sin)
        states = self.norm.forward(states)
        return states
    

class LlaMa2(Module):
    model: LlaMa2Model

    def __init__(self,
                embedding_size: int,
                nr_layers: int,
                hidden_size: int,
                nr_querys: int,
                nr_groups: int,
                head_size: int,
                internal_size: int
            ) -> None:
        super().__init__()
        self.model = self.register_module("model", LlaMa2Model(embedding_size, nr_layers, hidden_size, nr_querys, nr_groups, head_size, internal_size))
        self.lm_head = self.register_module("lm_head", Embedding(embedding_size, hidden_size))

    def forward(self, input_ids: Tensor):
        outputs = self.model.forward(input_ids)
        output_probs = self.lm_head.backward(outputs)
        return output_probs



def main():
    import os
    import torch
    import transformers
    path = "/home/ubuntu/Git/Qwen1.5-14B-Chat/"
    weights_paths = [
        os.path.join(path, f"model-0000{i+1}-of-00008.safetensors") for i in range(8)
    ]
    tokenizer_path = os.path.join(path, "tokenizer.json")
    # Qwen2-14B
    model = LlaMa2(
        embedding_size=152064,
        nr_layers=40,
        hidden_size=5120,
        nr_querys=40,
        nr_groups=1,
        head_size=128,
        internal_size=13696,
    )
    load_from_safetensors(model, weights_paths)
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

    prompt = "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n"
    while True:
        print("User:", end="\t", flush=True)
        line = input()
        prompt += f"<|im_start|>user\n {line}<|im_end|>\n<|im_start|>assistant\n"
        [input_ids,] = tokenizer([prompt]).input_ids
        result = ""
        print("Assistant:", end="\t", flush=True)
        while True:
            output_probs: numpy.ndarray = model.forward(CUDATensor.from_numpy(torch.tensor([input_ids], dtype=torch.int32))).numpy()
            output_id = output_probs.argmax(axis=-1)[0][-1].item()
            output_piece = tokenizer.decode(output_id)
            output_prob = output_probs[0][-1][output_id]
            # print(output_prob)
            result += output_piece
            if output_piece == "<|endoftext|>":
                raise
            if output_piece == "<|im_end|>":
                break
            print(output_piece, end="", flush=True)
            input_ids += [output_id]
        prompt += result + "<|im_end|>\n"
        print()


if __name__ == '__main__':
    main()

