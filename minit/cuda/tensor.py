import ctypes
from numbers import Number
from typing import Optional, Tuple

from .lib.cuda_runtime import load_cuda_runtime

from ..core.tensor import Tensor
from .allocator import CUDAMemory, copy_cuda_memory, sync_cuda

import numpy


class CUDATensor(Tensor):
    _memory: Optional[CUDAMemory] = None
    _shape: Tuple[int, ...]
    _item: Optional[Number] = None

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def data_ptr(self) -> int:
        return self.memory._pointer
    
    @property
    def memory(self) -> CUDAMemory:
        if self._memory is None and self._item is not None:
            self._memory = CUDAMemory(4)
            self.copy_from_numpy(numpy.full((), self._item, numpy.float32))
        return self._memory
            

    @staticmethod
    def allocate(shape: Tuple[int, ...], dtype: str) -> "CUDATensor":
        shape = tuple(shape)
        size = 4
        for dim in shape:
            size *= dim
        memory = CUDAMemory(size)
        result = CUDATensor()
        result._shape = shape
        result._memory = memory
        return result

    @staticmethod
    def wrap(shape: Tuple[int, ...], dtype: str, memory: CUDAMemory) -> "CUDATensor":
        shape = tuple(shape)
        size = 4
        for dim in shape:
            size *= dim
        assert memory._size == size
        result = CUDATensor()
        result._shape = shape
        result._memory = memory
        return result

    @staticmethod
    def from_numpy(array: numpy.ndarray):
        device_array = CUDATensor.allocate(array.shape, "float32")
        device_array.copy_from_numpy(array)
        return device_array

    @staticmethod
    def from_item(item: Number):
        result = CUDATensor()
        result._shape = ()
        result._item = item
        return result

    def copy_from_numpy(self, array: numpy.ndarray):
        assert array.shape == self.shape, f"{array.shape} vs {self.shape}"
        import torch
        if isinstance(array, torch.Tensor):
            array: torch.Tensor
            if array.dtype == torch.bfloat16:
                array = array.to(torch.float32)
        if array.dtype == numpy.half:
            array = array.astype(numpy.float32)
        array = numpy.ascontiguousarray(array)
        assert array.dtype == numpy.float32
        pointer, _read_only_flag = array.__array_interface__['data']
        size = 4
        for dim in array.shape:
            size *= dim
        sync_cuda()
        copy_cuda_memory(self.data_ptr, pointer, size)

    def numpy(self):
        host_data = numpy.ndarray(self.shape, "float32")
        pointer, _read_only_flag = host_data.__array_interface__['data']
        size = 4
        for dim in self.shape:
            size *= dim
        copy_cuda_memory(pointer, self.data_ptr, size)
        sync_cuda()
        return host_data

    def item(self):
        if self._item is not None:
            return self._item
        else:
            return self.numpy().item()
