import math
from typing import Dict, Tuple

import numpy
from minit.cuda.tensor import CUDATensor
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
import minit.quantize.dispatch
import nvtx

from minit.quantize.tensor import QuantizedTensor

default_dtype = "float16"

def debug_tensor(name: str, x: Tensor):
    print(f"{name} mean: {x.numpy().astype(numpy.float32).mean()} var: {x.numpy().astype(numpy.float32).var()}")

def dump_tensor(name: str, x: Tensor):
    numpy.save(f"{name}_minit", x.numpy())

def silu(x: Tensor) -> Tensor:
    return multiply(x, sigmoid(x))


def unpack(x: Tensor, axis: int) -> Tensor:
    assert x.dtype == "int32"
    last_dim = x.shape[-1]
    # NOTE: memory order matters
    x = x.reinterpret("uint8")
    x = x.expand(len(x.shape)-1, [last_dim, 32 // 8])
    x = x.add_axis(axis+1)
    x = x.transpose(axis+1, len(x.shape)-1)
    x = x.remove_axis(len(x.shape)-1)
    x = x.fold(axis, axis+2)
    return x



class QLinear(Module):
    qweight: Tensor
    qzeros: Tensor
    scales: Tensor
    g_idx: Tensor

    def __init__(self, in_features: int, out_features: int, nr_groups: int) -> None:
        super().__init__()
        self.qweight = self.register_buffer("qweight", (in_features//4, out_features), "int32")
        self.qzeros = self.register_buffer("qzeros", (nr_groups, out_features//4), "int32")
        self.scales = self.register_buffer("scales", (nr_groups, out_features), default_dtype)
        self.g_idx = self.register_buffer("g_idx", (in_features,), "int32")

    @property
    def weight(self):
        return QuantizedTensor(unpack(self.qweight, 0), self.g_idx, unpack(self.qzeros, 1), self.scales)

    def forward(self, x: Tensor):
        weight = self.weight.dequantize().transpose(0, 1)
        return matrix_multiply(x, weight)


class QLinearWithBias(Module):
    qweight: Tensor
    qzeros: Tensor
    scales: Tensor
    g_idx: Tensor
    bias: Tensor

    def __init__(self, in_features: int, out_features: int, nr_groups: int) -> None:
        super().__init__()
        self.qweight = self.register_buffer("qweight", (in_features//4, out_features), "int32")
        self.qzeros = self.register_buffer("qzeros", (nr_groups, out_features//4), "int32")
        self.scales = self.register_buffer("scales", (nr_groups, out_features), default_dtype)
        self.g_idx = self.register_buffer("g_idx", (in_features,), "int32")
        self.bias = self.register_buffer("bias", (out_features,), default_dtype)

    @property
    def weight(self):
        return QuantizedTensor(unpack(self.qweight, 0), self.g_idx, unpack(self.qzeros, 1), self.scales)

    def forward(self, x: Tensor):
        weight = self.weight.dequantize().transpose(0, 1)
        output = matrix_multiply(x, weight)
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
        self.c_attn = self.register_module("c_attn", QLinearWithBias(hidden_size, nr_querys * head_size + nr_key_values * head_size, 40))
        self.c_proj = self.register_module("c_proj", QLinearWithBias(nr_querys * head_size, hidden_size, 40))

    @nvtx.annotate("attention")
    def forward(self, x: Tensor, precomp_freqs_cos: Tensor, precomp_freqs_sin: Tensor):
        nr_key_values = (self.nr_querys // self.nr_groups) * 2
        assert len(x.shape) == 3
        qkv: Tensor = self.c_attn(x)
        q, k, v = qkv.split(2, [
            constant(self.nr_querys * self.head_size, "int32"),
            constant(nr_key_values // 2 * self.head_size, "int32"),
            constant(nr_key_values // 2 * self.head_size, "int32"),
        ])
        # b, s, h
        q = q.expand(2, [
            self.nr_querys,
            self.head_size,
        ])
        # b, s, nr_heads, head_size
        k = k.expand(2, [
            nr_key_values // 2,
            self.head_size,
        ])
        v = v.expand(2, [
            nr_key_values // 2,
            self.head_size,
        ])
        k = k.repeat_interleaved(2, self.nr_groups) # b, s, nr_heads, head_size
        v = v.repeat_interleaved(2, self.nr_groups)
        # b, s, nr_heads, head_size
        q = rope(q, precomp_freqs_cos, precomp_freqs_sin)
        k = rope(k, precomp_freqs_cos, precomp_freqs_sin)
        # b, s, nr_heads, head_size
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        # b, nr_heads, s, head_size
        qk = batch_matrix_multiply(q, k)
        # b, nr_heads, s, s
        qk = qk / math.sqrt(self.head_size) # b, nr_heads, s, s
        qk = qk + triangle_upper(fill(-math.inf, qk.shape, qk.dtype), diagonal=1)
        v = v.transpose(1, 2)
        v = v.transpose(2, 3)
        # b, nr_heads, head_size, s
        qk_softmax = softmax(qk, 3)
        output = batch_matrix_multiply(qk_softmax, v)
        # b, nr_heads, s, head_size
        output = output.transpose(1, 2)
        # b, s, nr_heads, head_size
        output = output.fold(2, 4)
        # b, s, h
        output = self.c_proj(output)
        return output


class FeedForward(Module):
    def __init__(self, hidden_size: int, internal_size: int) -> None:
        super().__init__()
        self.w1 = self.register_module("w1", QLinearWithBias(hidden_size, internal_size, 40))
        self.w2 = self.register_module("w2", QLinearWithBias(hidden_size, internal_size, 40))
        self.c_proj = self.register_module("c_proj", QLinearWithBias(internal_size, hidden_size, 107))

    @nvtx.annotate("feed_forward")
    def forward(self, x: Tensor):
        output0 = self.w1(x)
        output1 = silu(self.w2(x))
        output2 = self.c_proj(output0 * output1)
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
    attn: Attention
    mlp: FeedForward
    ln_1: RMSNorm
    ln_2: RMSNorm

    def __init__(self, hidden_size: int, nr_querys: int, nr_groups: int, head_size: int, internal_size: int) -> None:
        super().__init__()
        self.attn = self.register_module("attn", Attention(hidden_size, nr_querys, nr_groups, head_size))
        self.mlp = self.register_module("mlp", FeedForward(hidden_size, internal_size))
        self.ln_1 = self.register_module("ln_1", RMSNorm(hidden_size))
        self.ln_2 = self.register_module("ln_2", RMSNorm(hidden_size))

    @nvtx.annotate("layer")
    def forward(self, x: Tensor, precomp_freqs_cos: Tensor, precomp_freqs_sin: Tensor):
        attention_output = x + self.attn.forward(self.ln_1.forward(x), precomp_freqs_cos, precomp_freqs_sin)
        ffn_output = self.mlp.forward(self.ln_2.forward(attention_output))
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
    wte: Embedding
    h: ModuleList[TransformerBlock]
    freqs_cos: Tensor
    freqs_sin: Tensor
    ln_f: RMSNorm

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
        self.h = self.register_module("h", ModuleList())
        for _ in range(nr_layers):
            self.h.append(TransformerBlock(hidden_size, nr_querys, nr_groups, head_size, internal_size))
        self.wte = self.register_module("wte", Embedding(embedding_size, hidden_size))
        self.ln_f = self.register_module("ln_f", RMSNorm(hidden_size))
        self.freqs_cos, self.freqs_sin = self.precompute_freqs(head_size, 1024, 1e5)

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
        states = self.wte.forward(input_ids)
        freqs_cos = self.freqs_cos.slice(0, s, 0)
        freqs_sin = self.freqs_sin.slice(0, s, 0)
        all_states = []
        for _i, block in enumerate(self.h):
            all_states.append(states)
            block: TransformerBlock
            states = block.forward(states, freqs_cos, freqs_sin)
        states = self.ln_f.forward(states)
        return states
    

class LlaMa2(Module):
    transformer: LlaMa2Model

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
        self.transformer = self.register_module("transformer", LlaMa2Model(embedding_size, nr_layers, hidden_size, nr_querys, nr_groups, head_size, internal_size))
        self.lm_head = self.register_module("lm_head", Embedding(embedding_size, hidden_size))

    def forward(self, input_ids: Tensor):
        outputs = self.transformer.forward(input_ids)
        output_probs = self.lm_head.backward(outputs)
        return output_probs



def main():
    import os
    import sys
    import torch
    import transformers
    path = "/home/ubuntu/.cache/modelscope/hub/qwen/Qwen-14B-Chat-Int8"
    sys.path.append(os.path.join(path))
    from tokenization_qwen import QWenTokenizer
    weights_paths = [
        os.path.join(path, f"model-0000{i+1}-of-00008.safetensors") for i in range(8)
    ]
    tokenizer_path = os.path.join(path, "qwen.tiktoken")
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
    tokenizer = QWenTokenizer(tokenizer_path)

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

