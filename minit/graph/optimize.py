from typing import Any, Iterator, Optional, Tuple

from ..trace.function import trace_function, trace_function_on_graph
from . import ConstantNode, Graph, InternalNode, LinkedList, NodeRef, OperatorNodeRef, PlaceholderNode, ShapeNode, SubGraph, TensorNode, TensorNodeRef
from ..core.tensor import Tensor


class OptimizationPass:
    _pattern: Optional[SubGraph] = None

    def make_inputs(self) -> Tuple[Tensor, ...]:
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
    

class Matcher:
    pattern: SubGraph

    def __init__(self, pattern: SubGraph) -> None:
        self.pattern = pattern

    def match(self, target_graph: Graph, target_operators: LinkedList[OperatorNodeRef]) -> Optional[SubGraph]:
        pattern_nodes = {}
        target_nodes = {}
        target_nodes_rev = {}

        pattern_input_nodes = {}
        next_node_id = 0

        target_input_nodes = []
        target_output_nodes = []

        for pattern_input_ref in self.pattern.inputs:
            node_id = next_node_id
            next_node_id += 1
            pattern_nodes[pattern_input_ref] = node_id
            pattern_input_nodes[pattern_input_ref] = node_id

        def check_node(pattern_node_ref, target_node_ref) -> bool:
            assert isinstance(pattern_node_ref, NodeRef)
            assert isinstance(target_node_ref, NodeRef)
            # capture input
            if pattern_node_ref in pattern_input_nodes:
                if target_node_ref in target_nodes:
                    # disallow internal node as input for now
                    return None
                target_nodes[target_node_ref] = pattern_input_nodes[pattern_arg_ref]
                target_nodes_rev[pattern_input_nodes[pattern_arg_ref]] = target_arg_ref
                target_input_nodes.append(target_node_ref)
                return True
            if pattern_node_ref not in pattern_nodes:
                pattern_node = pattern_node_ref()
                target_node = target_node_ref()
                if isinstance(pattern_node, ConstantNode):
                    if not isinstance(target_node, ConstantNode):
                        return False
                    return pattern_node.value.item() == target_node.value.item()
                elif isinstance(pattern_node, ShapeNode):
                    if not isinstance(target_node, ShapeNode):
                        return False
                    return check_node(pattern_node.source().ref, target_node.source().ref)
                else:
                    assert False
            if not target_node_ref in target_nodes:
                return False
            return pattern_nodes[pattern_node_ref] == target_nodes[target_node_ref]
        
        if len(list(self.pattern.operators)) != len(list(target_operators)):
            return None
        
        target_operators_node = target_operators.head.next
        target_operators_tail = target_operators.tail
        pattern_operators_node = self.pattern.operators.head.next
        pattern_operators_tail = self.pattern.operators.tail

        while target_operators_node != target_operators_tail and pattern_operators_node != pattern_operators_tail:
            pattern_operator_ref = pattern_operators_node.value
            target_operator_ref = target_operators_node.value
            pattern_operator = pattern_operator_ref()
            target_operator = target_operator_ref()
            if pattern_operator.operator != target_operator.operator:
                # mismatch
                return None
            if len(pattern_operator.args) != len(target_operator.args):
                # mismatch
                return None
            for pattern_arg_ref, target_arg_ref in zip(pattern_operator.args, target_operator.args):
                if not check_node(pattern_arg_ref, target_arg_ref):
                    return None
            for pattern_output_use, target_output_use in zip(pattern_operator.outputs, target_operator.outputs):
                pattern_output_ref = pattern_output_use().ref
                target_output_ref = target_output_use().ref
                node_id = next_node_id
                next_node_id += 1
                pattern_nodes[pattern_output_ref] = node_id
                target_nodes[target_output_ref] = node_id
                target_nodes_rev[node_id] = target_output_ref
            pattern_operators_node = pattern_operators_node.next
            target_operators_node = target_operators_node.next

        if pattern_operators_node != pattern_operators_tail:
            return None

        for pattern_output_use in self.pattern.outputs:
            pattern_output_ref = pattern_output_use().ref
            assert pattern_output_ref in pattern_nodes
            target_output_nodes.append(target_nodes_rev[pattern_nodes[pattern_output_ref]]().use())

        result_graph = SubGraph(target_graph)
        result_graph.inputs = target_input_nodes
        result_graph.operators = target_operators
        result_graph.outputs = target_output_nodes
        return result_graph


class TraceGraphOptimizer:
    graph: SubGraph

    def __init__(self, graph: SubGraph) -> None:
        self.graph = graph

    def apply(self, optimization_pass: OptimizationPass):
        pattern = optimization_pass.pattern
        matcher = Matcher(pattern)
        operators = LinkedList()
        operators.head = self.graph.operators.head
        operators.tail = self.graph.operators.tail
        operators.check()
        while operators.head.next != operators.tail:
            result = matcher.match(self.graph.graph, operators)
            if result is not None:
                rewrite = optimization_pass.rewrite
                rewrite_result = trace_function_on_graph(rewrite, optimization_pass.make_inputs(), self.graph.graph, result.inputs)
                self.graph.replace(result, rewrite_result)
            else:
                operators.head = operators.head.next
                operators.check()
