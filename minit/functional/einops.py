from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union
import nvtx

from .linalg import batch_matrix_multiply, matrix_multiply
from .utils import _convert_scalar
from ..graph import SubGraph
from ..core.tensor import Tensor


__all__ = [
    "einsum",
    "rearrange",
]

ShapeTerm = Union[List[str], str]
ShapeSpec = List[ShapeTerm]
ShapeSpecs = List[ShapeSpec]


def parse_equation_term(term: str):
    dims = []
    i = 0
    while i < len(term):
        if term[i] =="(":
            i += 1
            subdims = []
            while term[i] != ")":
                assert term[i].isalpha()
                subdims.append(term[i])
                i += 1
            dims.append(subdims)
        elif term[i] == ".":
            assert term[i:i+3] == "..."
            i += 2
            dims.append("...")
        else:
            assert term[i].isalpha()
            dims.append(term[i])
        i += 1
    return dims


def serialize_spec(spec: ShapeSpec):
    term = ""
    for dim in spec:
        if isinstance(dim, list):
            term += "("
            for subdim in dim:
                term += subdim
            term += ")"
        else:
            term += dim
    return term


def flatten_spec(spec: ShapeSpec):
    flatten_spec = []
    for dim in spec:
        if isinstance(dim, list):
            dim: List[str]
            flatten_spec.extend(dim)
        else:
            dim: str
            flatten_spec.append(dim)
    return flatten_spec


def parse_equation(equation: str) -> List[ShapeSpecs]:
    [args, output] = equation.split("->")
    args = args.split(",")
    input_specs = [parse_equation_term(arg) for arg in args]
    output_spec = parse_equation_term(output)
    return [input_specs, [output_spec]]


def deduce_ellipsis(arg_specs: ShapeSpecs, args: Tuple[Tensor, ...]) -> Optional[List[str]]:
    ellipsis = None
    for arg_spec, arg in zip(arg_specs, args):
        for dim in arg_spec:
            if dim == "...":
                ellipsis_size = len(arg.shape) - (len(arg_spec)-1)
                assert ellipsis_size >= 0
                assert ellipsis is None
                ellipsis = [f".{i}" for i in range(ellipsis_size)]
    return ellipsis


def expand_ellipsis(specs: ShapeSpecs, ellipsis: Optional[List[str]]):
    if ellipsis is None:
        return
    count = 0
    for spec in specs:
        for i in range(len(spec)):
            if spec[i] == "...":
                new_spec = spec[:i] + ellipsis + spec[i+1:]
                spec.clear()
                spec.extend(new_spec)
                count += 1
    return count


@nvtx.annotate("einsum")
def einsum(equation: str, *args: Tensor, variables: Optional[Dict[str, Tensor]]=None):
    if variables is None:
        variables = {}
    input_specs, output_specs = parse_equation(equation)
    ellipsis = deduce_ellipsis(input_specs, args)
    expand_ellipsis(input_specs, ellipsis)
    expand_ellipsis(output_specs, ellipsis)
    [output_spec] = output_specs
    flatten_input_specs = list(map(flatten_spec, input_specs))
    flatten_output_spec = flatten_spec(output_spec)
    dim_counts = defaultdict(lambda: 0)
    for spec in [*flatten_input_specs[1:], flatten_output_spec]:
        for dim in spec:
            dim_counts[dim] += 1
    dims = flatten_input_specs[0]
    x = args[0]
    x_spec = input_specs[0]
    for i, (arg, input_spec, flatten_input_spec) in enumerate(zip(args[1:], input_specs[1:], flatten_input_specs[1:])):
        next_dims = flatten_input_spec
        bs = []
        ms = []
        ns = []
        ks = []
        for dim in next_dims:
            dim_counts[dim] -= 1
            count = dim_counts[dim]
            if dim in dims:
                if count > 0:
                    bs.append(dim)
                else:
                    ks.append(dim)
            else:
                ns.append(dim)
        for dim in dims:
            if dim not in next_dims:
                ms.append(dim)
        if len(bs) == 0:
            a_spec = [ms, ks]
            b_spec = [ns, ks]
            c_spec = [ms, ns]
            a = rearrange_impl(x_spec, a_spec, x, variables)
            b = rearrange_impl(input_spec, b_spec, arg, variables)
            c = matrix_multiply(a, b)
            x, x_spec = c, c_spec
        else:
            a_spec = [bs, ms, ks]
            b_spec = [bs, ns, ks]
            c_spec = [bs, ms, ns]
            a = rearrange_impl(x_spec, a_spec, x, variables)
            b = rearrange_impl(input_spec, b_spec, arg, variables)
            c = batch_matrix_multiply(a, b)
            x, x_spec = c, c_spec
    x = rearrange_impl(x_spec, output_spec, x, variables)
    return x


def rearrange_impl(input_spec, output_spec, x: Tensor, variables: Dict[str, Tensor]):
    # TODO: handle broadcast
    # expand axes
    for i, (dim, dim_size) in reversed(list(enumerate(zip(input_spec, x.shape)))):
        if isinstance(dim, list):
            subdim_sizes = []
            unknown_dim = None
            for subdim in dim:
                if subdim in variables:
                    subdim_sizes.append(variables[subdim])
                else:
                    subdim_sizes.append(None)
                    assert unknown_dim is None
                    unknown_dim = subdim
            if unknown_dim is not None:
                for subdim_size in subdim_sizes:
                    dim_size //= subdim_size
                variables[unknown_dim] = dim_size
                for i in range(len(subdim_sizes)):
                    if subdim_sizes[i] is None:
                        subdim_sizes[i] = dim_size
            x = x.expand(i, subdim_sizes)
        else:
            if dim not in variables:
                variables[dim] = dim_size
            else:
                # TODO: assertion?
                pass
    # shuffle axes
    flatten_input_spec = flatten_spec(input_spec)
    flatten_output_spec = flatten_spec(output_spec)
    for i, dim in enumerate(flatten_output_spec):
        if dim not in flatten_input_spec:
            flatten_input_spec.insert(i, dim)
            x = x.add_axis(i, variables[dim])
        if flatten_input_spec[i] == dim:
            continue
        j = flatten_input_spec.index(dim)
        flatten_input_spec[i], flatten_input_spec[j] = flatten_input_spec[j], flatten_input_spec[i]
        x = x.transpose(i, j)
    # fold axes
    for i, dim in enumerate(output_spec):
        if isinstance(dim, list):
            dim: List[str]
            x = x.fold(i, i+len(dim))
    return x


@nvtx.annotate("rearrange")
def rearrange(equation: str, x: Tensor, variables: Optional[Dict[str, Tensor]]=None) -> Tensor:
    if variables is None:
        variables = {}
    input_specs, output_specs = parse_equation(equation)
    ellipsis = deduce_ellipsis(input_specs, (x,))
    expand_ellipsis(input_specs, ellipsis)
    expand_ellipsis(output_specs, ellipsis)
    [output_spec] = output_specs
    assert len(input_specs) == 1
    input_spec = input_specs[0]
    return rearrange_impl(input_spec, output_spec, x, variables)
