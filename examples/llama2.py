import math
from typing import Tuple

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
from minit.module.checkpoint import load_from_torch
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
    def forward(self, x: Tensor, precomp_freqs_cos: Tensor, precomp_freqs_sin: Tensor):
        nr_key_values = (self.nr_querys // self.nr_groups) * 2
        assert len(x.shape) == 3
        q = self.wq(x)
        # b, s, h
        q = q.expand(2, [
            self.nr_querys,
            self.head_size,
        ])
        # b, s, nr_heads, head_size
        k = self.wk(x)
        k = k.expand(2, [
            nr_key_values // 2,
            self.head_size,
        ])
        v = self.wv(x)
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
        output = self.wo(output)
        return output


class FeedForward(Module):
    def __init__(self, hidden_size: int, internal_size: int) -> None:
        super().__init__()
        self.w1 = self.register_module("w1", Linear(hidden_size, internal_size))
        self.w2 = self.register_module("w2", Linear(internal_size, hidden_size))
        self.w3 = self.register_module("w3", Linear(hidden_size, internal_size))

    @nvtx.annotate("feed_forward")
    def forward(self, x: Tensor):
        output0 = silu(self.w1(x))
        output1 = self.w3(x)
        output2 = self.w2(output0 * output1)
        return output2


class RMSNorm(Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.weight = self.register_buffer("weight", (size,), default_dtype)

    @nvtx.annotate("rms_norm")
    def forward(self, x: Tensor):
        # weight = self.weight
        # while len(weight.shape) < len(x.shape):
        #     weight = weight.add_axis(0, x.shape[-1-len(weight.shape)])
        # axis = len(x.shape) - 1
        # return rms_norm(x, weight, axis, 1e-5) * weight
        axis = len(x.shape) - 1
        return rms_norm(x, self.weight, axis, 1e-5)


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
    def forward(self, x: Tensor, precomp_freqs_cos: Tensor, precomp_freqs_sin: Tensor):
        attention_output = x + self.attention.forward(self.attention_norm.forward(x), precomp_freqs_cos, precomp_freqs_sin)
        ffn_output = self.feed_forward.forward(self.ffn_norm.forward(attention_output))
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
        states = self.tok_embeddings.forward(input_ids)
        freqs_cos = self.freqs_cos.slice(0, s, 0)
        freqs_sin = self.freqs_sin.slice(0, s, 0)
        all_states = []
        for _i, block in enumerate(self.layers):
            all_states.append(states)
            block: TransformerBlock
            states = block.forward(states, freqs_cos, freqs_sin)
        states = self.norm.forward(states)
        output_probs = self.output.backward(states)
        return output_probs


def main():
    import os
    import torch
    path = "/home/ubuntu/Git/minit/Downloads/Mistral-7B-v0.2-Instruct/"
    weights_path = os.path.join(path, "consolidated.00.pth")
    tokenizer_path = os.path.join(path, "tokenizer.model")
    import sentencepiece
    model = LlaMa2(
        embedding_size=32000,
        nr_layers=32,
        hidden_size=4096,
        nr_querys=32,
        nr_groups=4,
        head_size=128,
        internal_size=14336,
    )
    load_from_torch(model, weights_path)
    tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)
    print(f"eos: {tokenizer.id_to_piece(tokenizer.eos_id())}")
    prompt = ""
    while True:
        print("User:", end="\t", flush=True)
        line = input()
        prompt += f"<s>[INST] {line} [/INST]"
        [input_ids,] = tokenizer.tokenize([
            prompt
        ])
        result = ""
        print("Assistant:", end="\t", flush=True)
        while True:
            output_probs: numpy.ndarray = model.forward(CUDATensor.from_numpy(torch.tensor([input_ids], dtype=torch.int32))).numpy()
            output_id = output_probs.argmax(axis=-1)[0][-1].item()
            output_piece = tokenizer.id_to_piece(output_id)
            output_prob = output_probs[0][-1][output_id]
            # print(output_prob)
            result += output_piece
            if output_piece == "<unk>":
                raise
            if output_piece == "</s>":
                break
            print(output_piece, end="", flush=True)
            input_ids += [output_id]
        prompt += (result + "<s>")
        print()


if __name__ == '__main__':
    main()

