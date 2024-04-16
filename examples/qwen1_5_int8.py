import ctypes
import functools
import math
from typing import Tuple

import numpy
from minit.compiler.cxx import CXXUnit
from minit.cuda.tensor import CUDATensor
import minit.functional as F
from minit.functional.generate import fill, generate_sequence
from minit.functional.special import rms_norm, rope, sigmoid, softmax
from minit.module import Module
from minit.core.tensor import Tensor
from minit.functional.arith import constant
from minit.functional.linalg import batch_matrix_multiply, matrix_multiply, triangle_upper
from minit.module.checkpoint import load_from_safetensors
from minit.module.list import ModuleList
from minit.compiler.nvcc import nvcc
from minit.cuda.kernel.utils import get_cuda_dtype
from minit.compiler.template import substitude
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
    return x * sigmoid(x)


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


@functools.lru_cache(maxsize=None)
def generate_quantized_matrix_multiply(name: str, dtype: str, output_dtype: str):
    kernel_name = f"qmm_{name}"
    kernel_template =\
"""
#include <cuda.h>
#include <cuda_fp16.h>
#include <stdexcept>


#define CUDA_ASSERT(expr)                                           \\
    do {                                                            \\
        auto _err = (expr);                                         \\
        if (_err != cudaSuccess) {                                  \\
            throw std::runtime_error(cudaGetErrorString(_err));     \\
        }                                                           \\
    } while (0)


#define CUDA_DEVICE_ASSERT(expr)                                            \\
    do {                                                                    \\
        auto _flag = (expr);                                                \\
        if (!_flag) {                                                       \\
            printf("assertion %s failed at %d\\n", #expr, (int)__LINE__);   \\
            __trap();                                                       \\
        }                                                                   \\
    } while (0)


static constexpr size_t char_bits = 8;

template <typename T>
struct Matrix {
    T* data;
    size_t nr_rows;
    size_t nr_cols;

    __device__ __forceinline__ T& operator()(size_t row, size_t col) {
        CUDA_DEVICE_ASSERT(row < nr_rows && col < nr_cols);
        return data[row * nr_cols + col];
    }
};


template <typename TPack, size_t nr_bits>
struct RowPackedMatrix {
    using pack_type = TPack;

    pack_type* data;
    size_t nr_rows;
    size_t nr_cols;

    static constexpr size_t pack_size = (sizeof(pack_type) * char_bits) / nr_bits;

    __device__ __forceinline__ pack_type operator()(size_t row, size_t col) const {
        CUDA_DEVICE_ASSERT(row < nr_rows && col < nr_cols);
        auto pack = data[row * (nr_cols / pack_size) + (col / pack_size)];
        pack_type mask = 0;
        for (size_t i = 0; i < nr_bits; ++i) {
            mask |= pack_type((pack_type)1 << i);
        }
        return (pack >> ((col % pack_size) * nr_bits)) & mask;
    }
};


template <typename TPack, size_t nr_bits>
struct ColPackedMatrix {
    using pack_type = TPack;

    pack_type* data;
    size_t nr_rows;
    size_t nr_cols;

    static constexpr size_t pack_size = (sizeof(pack_type) * char_bits) / nr_bits;

    __device__ __forceinline__ pack_type operator()(size_t row, size_t col) const {
        CUDA_DEVICE_ASSERT(row < nr_rows && col < nr_cols);
        auto pack = data[(row / pack_size) * nr_cols + col];
        pack_type mask = 0;
        for (size_t i = 0; i < nr_bits; ++i) {
            mask |= pack_type((pack_type)1 << i);
        }
        return (pack >> ((row % pack_size) * nr_bits)) & mask;
    }
};


using T = ${DATA_TYPE};
using TOut = ${OUTPUT_DATA_TYPE};
static constexpr size_t tile_nk = ${TILE_NK};
using pack_type = unsigned;
static_assert(sizeof(pack_type) == 4);

// thread + block together split on n
// block split on k
// no split m (for m)
// load shared_memory on inputs
// A[m, k] @ B[k, n] -> C[m, n]
// half      pack(k)
//
// zeros: [groups, n // 4]
// scales: [groups, n]
__global__ void kernel(T* x, pack_type* qweight, int* g_idx, pack_type* zeros, T* scales, TOut* outputs,
        int m, int n, int k, int nr_groups) {
    // for k:
    //      load n
    //      load m
    //      atomic_add()
    auto thread_index = threadIdx.x;
    auto block_x = blockIdx.x;
    auto block_y = blockIdx.y;
    CUDA_DEVICE_ASSERT(blockDim.x == tile_nk);
    // [1 x tile_nk]
    __shared__ T x_tile[tile_nk];
    float w_tile[tile_nk];
    // load weight
    auto m_offset = 0;
    auto n_offset = block_x * tile_nk + thread_index;
    auto k_offset = block_y * tile_nk;
    ColPackedMatrix<pack_type, 8> qweight_matrix {qweight, k, n};
    RowPackedMatrix<pack_type, 8> zeros_matrix {zeros, nr_groups, n};
    Matrix<T> scales_matrix {scales, nr_groups, n};
    Matrix<T> x_matrix {x, m, k};
    Matrix<TOut> outputs_matrix {outputs, m, n};
    for (int i = 0; i < tile_nk; ++i) {
        int index_k = k_offset + i;
        int index_n = n_offset;
        auto group = g_idx[index_k];
        float zero = zeros_matrix(group, n_offset) + 1;
        float scale = scales_matrix(group, n_offset);
        w_tile[i] = (float(qweight_matrix(index_k, n_offset)) - zero) * scale;
    }
    // load inputs
    for (int i = 0; i < m; ++i) {
        int index_m = i + m_offset;
        int index_k = k_offset + thread_index;
        x_tile[thread_index] = x_matrix(index_m, index_k);
        __syncthreads(); // acquire x_tile
        float result = 0;
        for (int j = 0; j < tile_nk; ++j) {
            result += (float)x_tile[j] * w_tile[j];
        }
        __syncthreads(); // release x_tile
        int index_n = n_offset;
        atomicAdd(&outputs_matrix(index_m, index_n), (TOut)result);
    }
}

extern "C" void ${KERNEL_NAME}(cudaStream_t stream, T* x, unsigned* qweight, int* g_idx, unsigned* zeros, T* scales, TOut* outputs,
        int m, int n, int k, int nr_groups) {
    auto nr_ntiles = (n + tile_nk - 1) / tile_nk;
    auto nr_ktiles = (k + tile_nk - 1) / tile_nk;
    if (nr_ntiles * tile_nk != n) {
        __builtin_trap();
    }
    if (nr_ktiles * tile_nk != k) {
        __builtin_trap();
    }
    dim3 grid_dim = {nr_ntiles, nr_ktiles};
    dim3 block_dim = {tile_nk};
    kernel<<<grid_dim, block_dim, 0, stream>>>(x, qweight, g_idx, zeros, scales, outputs, m, n, k, nr_groups);
    CUDA_ASSERT(cudaGetLastError());
}

"""
    kernel_source = substitude(kernel_template, {
        "DATA_TYPE": get_cuda_dtype(dtype),
        "OUTPUT_DATA_TYPE": get_cuda_dtype(output_dtype),
        "TILE_NK": "32",
        "KERNEL_NAME": kernel_name,
    })
    kernel = nvcc.compile(CXXUnit(entrance=kernel_name, source=kernel_source))
    return kernel


@nvtx.annotate("quantized_matrix_multiply")
def quantized_matrix_multiply(x: CUDATensor, qweight: CUDATensor, zeros: CUDATensor, scales: CUDATensor, groups: CUDATensor):
    kernel = generate_quantized_matrix_multiply("qmm", x.dtype, x.dtype)
    *ms, k0 = x.shape
    quad_k1, n0 = qweight.shape
    g0, quad_n1 = zeros.shape
    g1, n2 = scales.shape
    k2, = groups.shape
    assert k0.item() == quad_k1.item() * 4 == k2.item()
    assert g0.item() == g1.item()
    assert n0.item() == quad_n1.item() * 4 == n2.item()
    m_item = 1
    for m in ms:
        m_item *= m.item()
    k = k0
    n = n0
    g = g0
    assert scales.dtype == x.dtype
    output = CUDATensor.allocate((*ms, n0), x.dtype)
    output._memory.reset()
    kernel(None, ctypes.c_void_p(x.data_ptr), ctypes.c_void_p(qweight.data_ptr), ctypes.c_void_p(groups.data_ptr),
           ctypes.c_void_p(zeros.data_ptr), ctypes.c_void_p(scales.data_ptr), ctypes.c_void_p(output.data_ptr), 
           ctypes.c_int(m_item), ctypes.c_int(n.item()), ctypes.c_int(k.item()), ctypes.c_int(g.item()))
    return output


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
        output = quantized_matrix_multiply(x, self.qweight, self.qzeros, self.scales, self.g_idx)
        return output


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
        output = quantized_matrix_multiply(x, self.qweight, self.qzeros, self.scales, self.g_idx)
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
            self.nr_querys * self.head_size,
            nr_key_values // 2 * self.head_size,
            nr_key_values // 2 * self.head_size,
        ])
        q = q.rearrange("bs(nd)->bsnd", dict(
            n=self.nr_querys,
            d=self.head_size,
        ))
        k = k.rearrange("bs(nd)->bsnd", dict(
            n=nr_key_values // 2,
            d=self.head_size,
        ))
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
        variables = {}
        flatten_x = x.rearrange("bs->(bs)", variables)
        output = self.weight.index(flatten_x, 0)
        return output.rearrange("(bs)->bs", variables)

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
        t = generate_sequence(0, size, 1, dtype="float32")
        freqs = F.einsum("a,b->ab", t, freqs)
        freqs_cos = freqs.cosine()
        freqs_sin = freqs.sine()
        freqs_cos = freqs_cos.cast(default_dtype)
        freqs_sin = freqs_sin.cast(default_dtype)
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

