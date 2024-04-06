import math
from typing import Dict, Tuple

import numpy
from minit.cuda.tensor import CUDATensor
from minit.functional.generate import fill, generate_sequence
from minit.functional.shape import add_axis, broadcast, fold, expand, remove_axis, repeat, repeat_interleaved, transpose
from minit.functional.index import split, tie
from minit.functional.special import sigmoid
from minit.module import Module
from minit.core.tensor import Tensor
from minit.functional.arith import add, constant, cosine, divide, exponential, power, multiply, sine, square, square_root, subtract
from minit.functional.linalg import batch_matrix_multiply, matrix_multiply, triangle_lower, triangle_upper
from minit.functional.index import index, slice
from minit.functional.reduce import mean, sum, max
from minit.module.list import ModuleList
import minit.cuda.dispatch
import nvtx

def check_nan(x: Tensor):
    value = x.numpy()
    mean = value.mean()
    var = value.var()
    if mean > 1e20 or mean < 1e20:
        breakpoint()
    if var > 1e20:
        breakpoint()
    if numpy.any(numpy.isnan(value)):
        breakpoint()

def debug_tensor(name: str, x: Tensor):
    print(f"{name} mean: {x.numpy().mean()} var: {x.numpy().var()}")

def dump_tensor(name: str, x: Tensor):
    numpy.save(f"{name}_minit", x.numpy())

def silu(x: Tensor) -> Tensor:
    return multiply(x, sigmoid(x))

@nvtx.annotate("rope")
def rope(x: Tensor, precomp_freqs_cos: Tensor, precomp_freqs_sin: Tensor) -> Tensor:
    assert len(x.shape) == 4
    precomp_freqs_cos = broadcast(add_axis(precomp_freqs_cos, 0), 0, constant(x.shape[0]))
    precomp_freqs_cos = broadcast(add_axis(precomp_freqs_cos, 2), 2, constant(x.shape[2]))
    precomp_freqs_cos = add_axis(precomp_freqs_cos, 4)
    precomp_freqs_sin = broadcast(add_axis(precomp_freqs_sin, 0), 0, constant(x.shape[0]))
    precomp_freqs_sin = broadcast(add_axis(precomp_freqs_sin, 2), 2, constant(x.shape[2]))
    precomp_freqs_sin = add_axis(precomp_freqs_sin, 4)
    shape = x.shape
    head_size = shape[3]
    x_real, x_imag = split(expand(x, 3, [constant(head_size // 2), constant(2)]), 4, sizes=[constant(1), constant(1)])
    x_real, x_imag = subtract(multiply(x_real, precomp_freqs_cos), multiply(x_imag, precomp_freqs_sin)), add(multiply(x_imag, precomp_freqs_cos), multiply(x_real, precomp_freqs_sin))
    x = tie((x_real, x_imag), axis=4)
    x = fold(x, 3, 5)
    assert x.shape == shape
    return x

def rms_norm(x: Tensor, axis: int, eps: float) -> Tensor:
    output = divide(
        x,
        square_root(
            add(
                broadcast(
                    mean(
                        square(
                            x
                        ),
                        axis
                    ),
                    axis,
                    constant(x.shape[axis])
                ),
                fill(eps, x.shape, x.dtype),
            )
        ),
    )
    return output

@nvtx.annotate("softmax")
def softmax(x: Tensor, axis: int) -> Tensor:
    max_x = broadcast(max(x, axis), axis, constant(x.shape[axis]))
    x_sub_max_x = subtract(x, max_x)
    exp_x_sub_max_x = exponential(x_sub_max_x)
    result = divide(
        exp_x_sub_max_x,
        broadcast(sum(
            exp_x_sub_max_x,
            axis
        ), axis, constant(x.shape[axis]))
    )
    return result

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
        self.q_projection = self.register_buffer("q_projection", (nr_querys * head_size, hidden_size), "float32")
        self.k_projection = self.register_buffer("k_projection", (nr_key_values // 2 * head_size, hidden_size), "float32")
        self.v_projection = self.register_buffer("v_projection", (nr_key_values // 2 * head_size, hidden_size), "float32")
        self.output_projection = self.register_buffer("output_projection", (hidden_size, nr_querys * head_size), "float32")

    @nvtx.annotate("attention")
    def forward(self, x: Tensor, precomp_freqs_cos: Tensor, precomp_freqs_sin: Tensor):
        nr_key_values = (self.nr_querys // self.nr_groups) * 2
        assert len(x.shape) == 3
        q = matrix_multiply(x, self.q_projection)
        # b, s, h
        q = expand(q, 2, [
            constant(self.nr_querys),
            constant(self.head_size),
        ])
        # b, s, nr_heads, head_size
        k = matrix_multiply(x, self.k_projection)
        k = expand(k, 2, [
            constant(nr_key_values // 2),
            constant(self.head_size),
        ])
        v = matrix_multiply(x, self.v_projection)
        v = expand(v, 2, [
            constant(nr_key_values // 2),
            constant(self.head_size),
        ])
        k = repeat_interleaved(k, 2, constant(self.nr_groups)) # b, s, nr_heads, head_size
        v = repeat_interleaved(v, 2, constant(self.nr_groups))
        # b, s, nr_heads, head_size
        q = rope(q, precomp_freqs_cos, precomp_freqs_sin)
        k = rope(k, precomp_freqs_cos, precomp_freqs_sin)
        # b, s, nr_heads, head_size
        q = transpose(q, 1, 2)
        k = transpose(k, 1, 2)
        # b, nr_heads, s, head_size
        qk = batch_matrix_multiply(q, k)
        # b, nr_heads, s, s
        qk = divide(qk, fill(math.sqrt(self.head_size), qk.shape, qk.dtype)) # b, nr_heads, s, s
        qk = add(qk, triangle_upper(fill(-math.inf, qk.shape, qk.dtype), diagonal=1))
        v = transpose(v, 1, 2)
        v = transpose(v, 2, 3)
        # b, nr_heads, head_size, s
        qk_softmax = softmax(qk, 3)
        output = batch_matrix_multiply(qk_softmax, v)
        # b, nr_heads, s, head_size
        output = transpose(output, 1, 2)
        # b, s, nr_heads, head_size
        output = fold(output, 2, 4)
        # b, s, h
        output = matrix_multiply(output, self.output_projection)
        return output


class FeedForward(Module):
    def __init__(self, hidden_size: int, internal_size: int) -> None:
        super().__init__()
        self.weight0 = self.register_buffer("weight0", (internal_size, hidden_size), "float32")
        self.weight1 = self.register_buffer("weight1", (hidden_size, internal_size), "float32")
        self.weight2 = self.register_buffer("weight2", (internal_size, hidden_size), "float32")

    @nvtx.annotate("feed_forward")
    def forward(self, x: Tensor):
        output0 = silu(matrix_multiply(x, self.weight0))
        output1 = matrix_multiply(x, self.weight2)
        output2 = matrix_multiply(multiply(output0, output1), self.weight1)
        return output2
    

class RMSNorm(Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.weight = self.register_buffer("weight", (size,), "float32")

    @nvtx.annotate("rms_norm")
    def forward(self, x: Tensor):
        weight = self.weight
        while len(weight.shape) < len(x.shape):
            weight = broadcast(add_axis(weight, 0), 0, constant(x.shape[-1-len(weight.shape)]))
        axis = len(x.shape) - 1
        return multiply(rms_norm(x, axis, 1e-5), weight)


class TransformerBlock(Module):
    attention: Attention
    ffn: FeedForward
    attention_norm: RMSNorm
    ffn_norm: RMSNorm

    def __init__(self, hidden_size: int, nr_querys: int, nr_groups: int, head_size: int, internal_size: int) -> None:
        super().__init__()
        self.attention = self.register_module("attention", Attention(hidden_size, nr_querys, nr_groups, head_size))
        self.ffn = self.register_module("ffn", FeedForward(hidden_size, internal_size))
        self.attention_norm = self.register_module("attention_norm", RMSNorm(hidden_size))
        self.ffn_norm = self.register_module("ffn_norm", RMSNorm(hidden_size))

    @nvtx.annotate("layer")
    def forward(self, x: Tensor, precomp_freqs_cos: Tensor, precomp_freqs_sin: Tensor):
        attention_output = self.attention.forward(self.attention_norm.forward(x), precomp_freqs_cos, precomp_freqs_sin)
        ffn_output = self.ffn.forward(self.ffn_norm.forward(add(attention_output, x)))
        return add(add(attention_output, x), ffn_output)


class Embedding(Module):
    def __init__(self, size: int, dim: int) -> None:
        super().__init__()
        self.weight = self.register_buffer("weight", (size, dim), "int8")

    # [B, S] . [W, H] -> [B, S, H]
    @nvtx.annotate("embedding.forward")
    def forward(self, x: Tensor) -> Tensor:
        flatten_x = fold(x, 0, len(x.shape))
        output = index(self.weight, flatten_x, 0)
        return expand(output, 0, list(map(constant, x.shape)))

    # [B, S, H] . [W, H] -> [B, S, W]
    @nvtx.annotate("embedding.backward")
    def backward(self, x: Tensor) -> Tensor:
        return matrix_multiply(x, self.weight)

class LlaMa2(Module):
    input_embedding: Embedding
    layers: ModuleList[TransformerBlock]
    output_embedding: Embedding
    freqs_cos: Tensor
    freqs_sin: Tensor
    norm: RMSNorm

    def __init__(self, embedding_size: int, nr_layers: int, hidden_size: int, nr_querys: int, nr_groups: int, head_size: int, internal_size: int) -> None:
        super().__init__()
        self.layers = self.register_module("layers", ModuleList())
        for _ in range(nr_layers):
            self.layers.append(TransformerBlock(hidden_size, nr_querys, nr_groups, head_size, internal_size))
        self.input_embedding = self.register_module("input_embedding", Embedding(embedding_size, hidden_size))
        self.output_embedding = self.register_module("output_embedding", Embedding(embedding_size, hidden_size))
        self.norm = self.register_module("norm", RMSNorm(hidden_size))
        self.freqs_cos, self.freqs_sin = self.precompute_freqs(head_size, 1024, 1e6)

    def precompute_freqs(self, dim: int, size: int, theta: float) -> Tuple[Tensor, Tensor]:
        freqs = divide(fill(1.0, (dim//2,), "float32"), power(fill(theta, (dim//2,), "float32"), generate_sequence(constant(0), constant(dim//2), constant(2/dim))))
        freqs = add_axis(freqs, 1)
        t = generate_sequence(constant(0), constant(size), constant(1))
        t = add_axis(t, 1)
        freqs = matrix_multiply(t, freqs)
        return cosine(freqs), sine(freqs)

    @nvtx.annotate("llama")
    def forward(self, input_ids: Tensor):
        _b, s = input_ids.shape[:2]
        states = self.input_embedding.forward(input_ids)
        freqs_cos = slice(self.freqs_cos, constant(0), constant(s), 0)
        freqs_sin = slice(self.freqs_sin, constant(0), constant(s), 0)
        all_states = []
        for _i, block in enumerate(self.layers):
            all_states.append(states)
            block: TransformerBlock
            states = block.forward(states, freqs_cos, freqs_sin)
        states = self.norm.forward(states)
        output_probs = self.output_embedding.backward(states)
        return output_probs
    

def load_mistral_model(ckpt: Dict[str, numpy.ndarray]):
    model = LlaMa2(
        embedding_size=32000,
        nr_layers=32,
        hidden_size=4096,
        nr_querys=32,
        nr_groups=4,
        head_size=128,
        internal_size=14336,
    )
    for name, array in ckpt.items():
        print(f"loading weight {name} {array.shape}")
        if name.startswith("layers."):
            layer_id = int(name.split(".")[1])
        else:
            layer_id = None
        if name == "tok_embeddings.weight":
            model.input_embedding.weight.copy_from_numpy(array)
        elif name == "norm.weight":
            model.norm.weight.copy_from_numpy(array)
        elif name == "output.weight":
            model.output_embedding.weight.copy_from_numpy(array)
        elif name.endswith(".attention.wq.weight"):
            model.layers[layer_id].attention.q_projection.copy_from_numpy(array)
        elif name.endswith(".attention.wk.weight"):
            model.layers[layer_id].attention.k_projection.copy_from_numpy(array)
        elif name.endswith(".attention.wv.weight"):
            model.layers[layer_id].attention.v_projection.copy_from_numpy(array)
        elif name.endswith(".attention.wo.weight"):
            model.layers[layer_id].attention.output_projection.copy_from_numpy(array)
        elif name.endswith(".feed_forward.w1.weight"):
            model.layers[layer_id].ffn.weight0.copy_from_numpy(array)
        elif name.endswith(".feed_forward.w2.weight"):
            model.layers[layer_id].ffn.weight1.copy_from_numpy(array)
        elif name.endswith(".feed_forward.w3.weight"):
            model.layers[layer_id].ffn.weight2.copy_from_numpy(array)
        elif name.endswith(".attention_norm.weight"):
            model.layers[layer_id].attention_norm.weight.copy_from_numpy(array)
        elif name.endswith(".ffn_norm.weight"):
            model.layers[layer_id].ffn_norm.weight.copy_from_numpy(array)
        else:
            assert False
    return model


def main():
    weights_path = "/home/ubuntu/Git/minit/Downloads/Mistral-7B-v0.2-Instruct/consolidated.00.pth"
    tokenizer_path = "/home/ubuntu/Git/minit/Downloads/Mistral-7B-v0.2-Instruct/tokenizer.model"
    ckpt = {}
    import torch
    print(f"loading checkpoint from {weights_path}")
    ckpt = torch.load(weights_path)
    import sentencepiece
    model = load_mistral_model(ckpt)
    tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)
    print(f"eos: {tokenizer.id_to_piece(tokenizer.eos_id())}")
    prompt = ""
    while True:
        print("User:", end="", flush=True)
        line = input()
        prompt += f"<s>[INST] {line} [/INST]"
        [input_ids,] = tokenizer.tokenize([
            prompt
        ])
        result = ""
        print("Assistant:", end="", flush=True)
        while True:
            output_probs: numpy.ndarray = model.forward(CUDATensor.from_numpy(numpy.array([input_ids], dtype=numpy.float32))).numpy()
            output_id = output_probs.argmax(axis=-1)[0][-1].item()
            output_piece = tokenizer.id_to_piece(output_id)
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

