import ctypes
from numbers import Number
from typing import Optional, Tuple

from ..core.shape import to_immediate_shape
from ..core.scalar import ScalarTensor
from ..core.dtype import dtype_info
from .lib.cuda_runtime import load_cuda_runtime
from ..core.tensor import Tensor
from .allocator import CUDAMemory, copy_cuda_memory, sync_cuda

import numpy


class CUDATensor(Tensor):
    _memory: Optional[CUDAMemory] = None
    _shape: Tuple[int, ...]
    _item: Optional[Number] = None
    _dtype: str

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(map(lambda x: ScalarTensor(x, "int32"), self._shape))

    @property
    def dtype(self) -> str:
        return self._dtype

    @property
    def data_ptr(self) -> int:
        return self.memory._pointer
    
    @property
    def memory(self) -> CUDAMemory:
        if self._memory is None and self._item is not None:
            self._memory = CUDAMemory(dtype_info(self.dtype).size_in_bytes)
            self.copy_from_numpy(numpy.full((), self._item, getattr(numpy, self.dtype)))
        return self._memory

    @staticmethod
    def allocate(shape: Tuple[int, ...], dtype: str) -> "CUDATensor":
        shape = to_immediate_shape(shape)
        size = dtype_info(dtype).size_in_bytes
        for dim in shape:
            assert not isinstance(dim, Tensor)
            size *= dim
        memory = CUDAMemory(size)
        result = CUDATensor()
        result._shape = shape
        result._memory = memory
        result._dtype = dtype
        return result

    @staticmethod
    def wrap(shape: Tuple[int, ...], dtype: str, memory: CUDAMemory) -> "CUDATensor":
        shape = to_immediate_shape(shape)
        size = dtype_info(dtype).size_in_bytes
        for dim in shape:
            size *= dim
        assert memory._size == size
        result = CUDATensor()
        result._shape = shape
        result._memory = memory
        result._dtype = dtype
        return result

    @staticmethod
    def from_numpy(array: numpy.ndarray):
        dtype = str(array.dtype).split(".")[-1]
        device_array = CUDATensor.allocate(array.shape, dtype)
        device_array.copy_from_numpy(array)
        return device_array

    @staticmethod
    def from_item(item: Number, dtype: str):
        result = CUDATensor()
        result._shape = ()
        result._item = item
        result._dtype = dtype
        return result

    def copy_from_numpy(self, array: numpy.ndarray):
        assert array.shape == self._shape, f"{array.shape} vs {self._shape}"
        import torch
        if isinstance(array, torch.Tensor):
            array: torch.Tensor
            if not array.is_cpu:
                array = array.cpu()
            array = array.contiguous()
            pointer = array.data_ptr()
        else:
            array: numpy.ndarray
            array = numpy.ascontiguousarray(array)
            pointer, _read_only_flag = array.__array_interface__['data']
        dtype = str(array.dtype).split(".")[-1]
        assert dtype == self.dtype
        size = dtype_info(dtype).size_in_bytes
        for dim in array.shape:
            size *= dim
        sync_cuda()
        copy_cuda_memory(self.data_ptr, pointer, size)

    def numpy(self):
        host_data = numpy.full(self._shape, 0, self.dtype)
        host_data = numpy.ascontiguousarray(host_data)
        pointer, _read_only_flag = host_data.__array_interface__['data']
        size = dtype_info(self.dtype).size_in_bytes
        for dim in self._shape:
            size *= dim
        copy_cuda_memory(pointer, self.data_ptr, size)
        sync_cuda()
        return host_data

    def item(self):
        if self._item is not None:
            return dtype_info(self.dtype).python_type(self._item)
        else:
            return self.numpy().item()
