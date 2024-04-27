import os
from typing import Callable, Dict, List
import json

from ..core.torch import TorchTensor
from ..core.tensor import Tensor
from ..core.shape import to_immediate_shape
from .module import Module


def load_from_torch(model: Module, path: str):
    import torch
    print(f"loading checkpoint from {path}")
    checkpoint: Dict[str, torch.Tensor] = torch.load(path)
    for name, array in checkpoint.items():
        print(f"loading checkpoint {name}")
        dtype = model.get_buffer(name).dtype
        array = array.to(getattr(torch, dtype))
        model.update_buffer(name, TorchTensor(array))
    return model


def load_from_safetensors(model: Module, paths: List[str], epilogue: Callable[[str, Tensor], Tensor] = lambda x: x):
    import torch
    import safetensors
    for path in paths:
        print(f"loading safetensors from {path}")
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                array = f.get_tensor(key)
                dtype = model.get_buffer(key).dtype
                print(f"loading safetensor {key} {array.shape} {array.dtype} -> {dtype}")
                shape = to_immediate_shape(model.get_buffer(key).shape)
                array = array.to(getattr(torch, dtype))
                assert array.shape == shape, f"{array.shape} vs {shape}"
                model.update_buffer(key, epilogue(key, TorchTensor(array)))
    return model


def load_from_safetensors_index(model: Module, path: str, epilogue: Callable[[str, Tensor], Tensor] = lambda k, v: v):
    with open(path) as f:
        index = json.load(f)
    parts = list(map(lambda part: os.path.join(os.path.dirname(path), part), dict.fromkeys(index["weight_map"].values())))
    load_from_safetensors(model, parts, epilogue)
