from typing import Dict, Generic, List, Optional, Sequence, Set, Tuple, TypeVar

from ..trace.function import trace_function_on_graph
from . import Graph, GraphBuilder, LinkedListIterator, LinkedListView, OperatorNode, ShapeUse, SubGraph, TensorNode, ValueUse
from ..core.tensor import Tensor


class OptimizationPass:
    _pattern: Optional[SubGraph] = None

    def make_inputs(self) -> Sequence[Tensor]:
        raise NotImplementedError()

    def describe(self, *inputs: Tensor) -> Tuple[Tensor, ...]:
        raise NotImplementedError()

    def predicate(self, *inputs: Tensor) -> bool:
        raise NotImplementedError()

    def rewrite(self, *inputs: Tensor) -> Tuple[Tensor, ...]:
        raise NotImplementedError()

    @property
    def pattern(self) -> SubGraph:
        from ..trace.function import trace_function
        if self._pattern is not None:
            return self._pattern
        inputs = self.make_inputs()
        self._pattern = trace_function(self.describe, inputs)
        return self._pattern


_T = TypeVar("_T")


class ListIterator(Generic[_T]):
    items: List[_T]
    index: int = 0

    def __init__(self, items: List[_T]):
        self.items = items

    def __next__(self):
        if self.index >= len(self.items):
            raise StopIteration
        value = self.items[self.index]
        self.index += 1
        return value

    def __iter__(self):
        return self


class Matcher:
    pattern: Graph

    def __init__(self, pattern: Graph) -> None:
        self.pattern = pattern

    def match(self, graph: Graph, operators_head: LinkedListIterator[OperatorNode]) -> Optional[SubGraph]:
        pattern = self.pattern
        nr_inputs = len(pattern.inputs)
        operators = operators_head.clone()
        variables = {}
        pattern_variables = {}
        for i, input in enumerate(pattern.inputs):
            pattern_variables[input] = i
        variable_id = nr_inputs
        users: Set[OperatorNode] = set()
        id2variable: Dict[int, TensorNode] = {}
        pattern_outputs = set(pattern_output.target for pattern_output in pattern.outputs)
        for i, pattern_operator in enumerate(pattern.operators):
            try:
                operator = next(operators)
            except StopIteration:
                return None
            if operator.operator != pattern_operator.operator:
                return None
            if len(operator.args) != len(pattern_operator.args):
                return None
            for arg, pattern_arg in zip(operator.args, pattern_operator.args):
                if arg.axis != pattern_arg.axis:
                    return None
                # capture input
                arg_id = pattern_variables[pattern_arg.target]
                if arg.target not in variables:
                    if arg_id >= nr_inputs:
                        return None
                    variables[arg.target] = arg_id
                    id2variable[arg_id] = arg.target
                if variables[arg.target] != arg_id:
                    return None
            if operator in users:
                users.remove(operator)
            for output, pattern_output in zip(operator.outputs, pattern_operator.outputs):
                variables[output] = variable_id
                id2variable[variable_id] = output
                pattern_variables[pattern_output] = variable_id
                if pattern_output not in pattern_outputs:
                    for use in output.uses:
                        user = use().user
                        if user is not None:
                            assert user != operator
                            users.add(user)
                variable_id += 1
        if len(users) != 0:
            for user in users:
                assert user.valid
            return None
        inputs = []
        outputs = []
        for pattern_input in pattern.inputs:
            inputs.append(id2variable[pattern_variables[pattern_input]].use_value(None))
        for pattern_output in pattern.outputs:
            if isinstance(pattern_output, ShapeUse):
                outputs.append(id2variable[pattern_variables[pattern_output.target]].use_shape(None, pattern_output.axis))
            elif isinstance(pattern_output, ValueUse):
                outputs.append(id2variable[pattern_variables[pattern_output.target]].use_value(None))
            else:
                assert False
        return SubGraph(graph, inputs, LinkedListView(operators_head.current, operators.current), outputs)


class TraceGraphOptimizer:
    graph: SubGraph

    def __init__(self, graph: SubGraph) -> None:
        self.graph = graph

    def apply(self, optimization_pass: OptimizationPass) -> int:
        pattern = optimization_pass.pattern
        matcher = Matcher(pattern)
        operators = iter(self.graph.operators)
        count = 0
        while True:
            result = matcher.match(self.graph, operators)
            if result is not None:
                count += 1
                rewrite = optimization_pass.rewrite
                old_operators = result.operators.tolist()
                for operator in old_operators:
                    operator.destroy()
                result.operators.clear()
                builder = GraphBuilder(self.graph.graph, result.inputs, result.operators)
                rewrite_outputs = trace_function_on_graph(rewrite, optimization_pass.make_inputs(), builder, result.inputs)
                for output, rewrite_output in zip(result.outputs, rewrite_outputs, strict=True):
                    assert output.axis == rewrite_output.axis
                    output.target.replace(rewrite_output.target)
            else:
                try:
                    next(operators)
                except StopIteration:
                    break
        return count
