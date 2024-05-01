import nvtx

from .tensor import NCCLTensor
from ..distributed.group import get_world
from ..cuda.tensor import CUDATensor
from ..core.tensor import Tensor
from ..core.dispatch import register_dispatch
from ..distributed.operator import DistributedBroadcast, DistributedAllReduce, DistributedAllGather
from .kernel import _generate_nccl_primitives

@register_dispatch()
def dispatch_all_gather(op: DistributedAllGather, version: NCCLTensor, x: CUDATensor):
    if op.axis != 0:
        x = x.transpose(0, op.axis)
    shape = x._shape
    size = 1
    for dim in shape:
        size *= dim
    shape = ((shape[0] * get_world().size),) + shape[1:]
    z = CUDATensor.allocate(shape, x._dtype)
    all_gather, _, _ = _generate_nccl_primitives(x._dtype)
    with nvtx.annotate(f"all_gather_{size}_{x._dtype}"):
        all_gather(None, version.comm, x.data_ptr, z.data_ptr, size)
    if op.axis != 0:
        z = z.transpose(0, op.axis)
    return (version, z,)

@register_dispatch()
def dispatch_all_reduce(op: DistributedAllReduce, version: NCCLTensor, x: CUDATensor):
    shape = x._shape
    size = 1
    for dim in shape:
        size *= dim
    z = CUDATensor.allocate(shape, x._dtype)
    _, all_reduce, _ = _generate_nccl_primitives(x._dtype)
    with nvtx.annotate(f"all_reduce_{size}_{x._dtype}"):
        all_reduce(None, version.comm, x.data_ptr, z.data_ptr, size)
    return (version, z,)

@register_dispatch()
def dispatch_broadcast(op: DistributedBroadcast, version: NCCLTensor, x: CUDATensor):
    shape = x._shape
    size = 1
    for dim in shape:
        size *= dim
    z = CUDATensor.allocate(shape, x._dtype)
    _, _, broadcast = _generate_nccl_primitives(x._dtype)
    with nvtx.annotate(f"broadcast_{size}_{x._dtype}"):
        broadcast(None, version.comm, x.data_ptr, z.data_ptr, size, op.source)
    return (version, z,)
