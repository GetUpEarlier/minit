from typing import Dict, List

from ..cuda.tensor import CUDATensor
from .module import Module


def load_from_torch(model: Module, path: str):
    import torch
    print(f"loading checkpoint from {path}")
    checkpoint: Dict[str, torch.Tensor] = torch.load(path)
    for name, array in checkpoint.items():
        print(f"loading checkpoint {name}")
        dtype = model.get_buffer(name).dtype
        array = array.to(getattr(torch, dtype))
        model.update_buffer(name, CUDATensor.from_numpy(array))
    return model


def load_from_safetensors(model: Module, paths: List[str]):
    import torch
    import safetensors
    for path in paths:
        print(f"loading safetensors from {path}")
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                print(f"loading safetensor {key}")
                array = f.get_tensor(key)
                dtype = model.get_buffer(key).dtype
                array = array.to(getattr(torch, dtype))
                model.update_buffer(key, CUDATensor.from_numpy(array))
    return model
