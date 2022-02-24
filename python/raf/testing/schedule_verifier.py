# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schedule verifier."""
# pylint: disable=too-many-instance-attributes
from collections import defaultdict
from typing import List, Mapping, Union
import tvm
import tvm.relay
from tvm.relay import ExprVisitor
from tvm.relay.analysis import free_vars
from tvm.relay import Expr, Var
from raf._ffi.ir.constant import ExtractValue


STREAM_OPS = {
    "set_stream": tvm.ir.op.Op.get("raf.op.set_stream"),
    "add_event": tvm.ir.op.Op.get("raf.op.add_event"),
    "wait_event": tvm.ir.op.Op.get("raf.op.wait_event"),
    "stream_barrier": tvm.ir.op.Op.get("raf.op.stream_barrier"),
}


class ExecutionOrderError(Exception):
    """ANF execution order error."""

    def __init__(self, expr_a, expr_b):
        msg = f"{expr_a} must be executed before {expr_b}, but their execution may overlap."
        super().__init__(self, msg)
        self.expr_a = expr_a
        self.expr_b = expr_b


def flatten_a_normal_form(e):
    """
    Flatten a relay expression in ANF form into a sequence of let values and a map from var to let
    value. Raise ValueError if it is not in ANF.

    Parameters
    ----------
    e : tvm.relay.Expr
        The relay expression to be checked.

    Returns
    -------
    nodes, var2value:
        nodes : List[tvm.relay.Expr]
            The list contains the let values and the innermost let body.
        var2value : Map[tvm.relay.Var, tvm.relay.Expr]
            The mapping from the let var to its binding expression.

    Raises
    ------
    ValueError:
        Indicate given expression is not in ANF.
    NotImplementedError:
        Given expression contains some sub-expression for which the we have not implemented for.
    """

    class ANFValueChecker(ExprVisitor):
        """
        Check whether an expression is a valid let value in ANF. It is valid if all its
        sub-expressions are valid sub-expression. A valid sub-expression is valid if it is Var,
        GlobalVar, Constant, Op, or primitive Function.
        """

        # pylint: disable=missing-function-docstring

        def __init__(self):
            super().__init__()
            # the non atomic expr we have met
            self.let_value = None

        @staticmethod
        def is_valid_sub_expr(expr):
            """
            Check whether given expression is a valid sub-expression in a let value in ANF.

            Parameters
            ----------
            expr : tvm.relay.Expr
                The given expression.

            Returns
            -------
            ret : bool
                Whether the given expr is a valid sub_expr
            """
            if isinstance(
                expr, (tvm.relay.Var, tvm.relay.GlobalVar, tvm.relay.Constant, tvm.ir.Op)
            ):
                return True
            if isinstance(expr, tvm.relay.Function):
                attrs = expr.attrs
                if attrs and "Primitive" in attrs and attrs["Primitive"] == 1:
                    return True
            return False

        def visit(self, expr):
            if self.let_value is None:
                self.let_value = expr
            else:
                if not self.is_valid_sub_expr(expr):
                    raise ValueError(f"Let value {self.let_value} referred {expr}, violating ANF.")
            super().visit(expr)

        def visit_function(self, fn):
            # We take Function as a constant, thus do not visit its sub-expressions
            return

        def visit_if(self, i):
            raise NotImplementedError(f"Support for type {type(i)} is not implemented yet.")

        def visit_let(self, let):
            raise NotImplementedError(f"Support for type {type(let)} is not implemented yet.")

    var2value = {}
    while isinstance(e, tvm.relay.Let):
        var2value[e.var] = e.value
        ANFValueChecker().visit(e.value)
        e = e.body
    ANFValueChecker().visit(e)
    return list(var2value.values()) + [e], var2value


class DependencyGraph:
    """
    Graph that supports reachability checking.
    """

    def __init__(self, nodes):
        """
        Init the graph with nodes.

        Parameters
        ----------
        nodes : List[tvm.relay.Expr]
            The nodes in the graph.
        """
        self.nodes = nodes
        self.children = defaultdict(list)
        self.parents = defaultdict(list)
        self.can_reach = None

    def add_edge(self, node_u: Expr, node_v: Expr):
        """
        Add an edge from node u to node v to the graph.

        Parameters
        ----------
        node_u : tvm.relay.Expr
            The source node.
        node_v : tvm.relay.Expr
            The dest node.
        """
        self.children[node_u].append(node_v)
        self.parents[node_v].append(node_u)

    def init_reachability_checking(self):
        """
        Initialize the reachability status for checking. Please call this function before calling
        `check_reach`.
        """
        out_degree = defaultdict(int)
        for node_u in self.nodes:
            out_degree[node_u] = len(self.children[node_u])
        stack = []
        for node_u in self.nodes:
            if out_degree[node_u] == 0:
                stack.append(node_u)
        order = []
        while len(stack) != 0:
            node_u = stack.pop()
            order.append(node_u)
            for parent in self.parents[node_u]:
                out_degree[parent] -= 1
                if out_degree[parent] == 0:
                    stack.append(parent)
        self.can_reach = defaultdict(set)
        for node_u in order:
            self.can_reach[node_u] = set(self.children[node_u])
            for node_v in self.children[node_u]:
                self.can_reach[node_u].update(self.can_reach[node_v])

    def check_reach(self, node_u, node_v):
        """
        Check whether node u can reach node v through a path in the graph. Please call
        `init_reachability_checking` before calling this function.

        Parameters
        ----------
        node_u : tvm.relay.Expr
            The source node.
        node_v : tvm.relay.Expr
            The dest node.

        Returns
        -------
        ret : bool
            Whether there exists a path from node u to node v in this graph.
        """
        assert self.can_reach, "Call init_reachability_checking first before call check_reach"
        return node_v in self.can_reach[node_u]


def build_dataflow_graph(nodes: List[Expr], var2value: Mapping[Var, Expr]):
    """
    Build the dataflow graph.

    Parameters
    ----------
    nodes : List[tvm.relay.Expr]
        The let values and innermost let body.

    var2value : Mapping[tvm.relay.Var, tvm.relay.Expr]
        The mapping from let var to its bind value.

    Returns
    -------
    graph : DependencyGraph
        The dataflow graph.
    """
    graph = DependencyGraph(nodes)
    for node_v in nodes:
        for var in free_vars(node_v):
            if var in var2value:
                node_u = var2value[var]
                graph.add_edge(node_u, node_v)
    return graph


def build_control_graph(nodes: List[Expr]):
    # pylint: disable=too-many-branches
    """
    Build the control graph.

    Parameters
    ----------
    nodes : List[tvm.relay.Expr]
        The let values and innermost let body.

    var2value : Mapping[tvm.relay.Var, tvm.relay.Expr]
        The mapping from let var to its bind value.

    Returns
    -------
    graph : DependencyGraph
        The control graph.
    """
    # The mapping from stream_id to the sequence of nodes executed on that stream
    stream_nodes = defaultdict(list)
    # The mapping from event_id to the add_event node
    id2event = {}
    # The current stream_id, -1 means default stream.
    current_stream_id = -1

    # Classify the nodes into corresponding stream_nodes. Record the the mapping from event_id to
    # add_event node.
    previous_barrier = None
    for node in nodes:
        if isinstance(node, tvm.relay.Call):
            if node.op == STREAM_OPS["set_stream"]:
                stream_id = ExtractValue(node.args[1]).value
                current_stream_id = stream_id
            elif node.op == STREAM_OPS["add_event"]:
                event_id = ExtractValue(node.args[0]).value
                id2event[event_id] = node
            elif node.op == STREAM_OPS["wait_event"]:
                event_id = ExtractValue(node.args[0]).value
                if event_id not in id2event:
                    raise ValueError(f"Event {event_id} used before add.")
            elif node.op == STREAM_OPS["stream_barrier"]:
                for stream_id in stream_nodes:
                    if current_stream_id == stream_id:
                        # avoid add barrier to this stream twice
                        continue
                    stream_nodes[stream_id].append(node)
                previous_barrier = node
        if current_stream_id not in stream_nodes and previous_barrier:
            stream_nodes[current_stream_id].append(previous_barrier)
        stream_nodes[current_stream_id].append(node)

    # Create the control graph
    graph = DependencyGraph(nodes)
    # Add the control edges from add_event node to wait_event node with the same event_id
    for stream_id in stream_nodes:
        for node in stream_nodes[stream_id]:
            if isinstance(node, tvm.relay.Call) and node.op == STREAM_OPS["wait_event"]:
                event_id = ExtractValue(node.args[0]).value
                event_node = id2event[event_id]
                wait_node = node
                graph.add_edge(event_node, wait_node)

    # Add the control edges between consecutive nodes on the same stream
    for stream_id in stream_nodes:
        for i in range(len(stream_nodes[stream_id]) - 1):
            graph.add_edge(stream_nodes[stream_id][i], stream_nodes[stream_id][i + 1])

    return graph


def check_data_dependency(dataflow_graph, control_graph):
    """
    Check whether all dependencies in dataflow graph have been satisfied in control_graph, i.e., the
    transitive closure of dataflow graph is a subgraph of the transitive closure of control graph.

    Parameters
    ----------
    dataflow_graph : DependencyGraph
        The dataflow graph.

    control_graph : DependencyGraph
        The control graph.

    Raises
    ------
    ExecutionOrderError:
        Raise this error when there is a dependency in dataflow graph is not satisfied in the
        control graph.
    """
    control_graph.init_reachability_checking()
    for node_u in dataflow_graph.nodes:
        for node_v in dataflow_graph.children[node_u]:
            if not control_graph.check_reach(node_u, node_v):
                raise ExecutionOrderError(node_u, node_v)


def verify_expr_schedule(e):
    """
    This function verifies whether a relay expression is correctly scheduled. The correctly
    scheduled relay program should satisfies the following conditions:
        1. In ANF form.
        2. The inputs of a node must be ready to use when we execute that node.

    To check the second condition, we construct two graphs for given relay program: control graph
    and dataflow graph. The nodes in the two graphs are the let values and the innermost body of
    the let chain. In control graph, an edge from node u to node v means that node u would be
    executed before node v. In dataflow graph, an edge from node u to node v means that node v
    would use the output of node u. We check, for each edge from node u to node v in dataflow
    graph, whether there exists a path from node u to node v.
    """
    nodes, var2value = flatten_a_normal_form(e)
    dataflow_graph = build_dataflow_graph(nodes, var2value)
    control_graph = build_control_graph(nodes)
    check_data_dependency(dataflow_graph, control_graph)


def verify_schedule(mod_or_func_or_expr: Union[tvm.IRModule, tvm.relay.Function, tvm.relay.Expr]):
    """
    Verify whether a scheduled function or relay expression is correct. When a tvm.ir.IRModule is
    given, its 'main' function is used to verify. The given function or expression must be in
    A-Normal Form.

    The correctly scheduled relay program should satisfies the following conditions:
        1. In ANF form.
        2. The inputs of a node must be ready to use when we execute that node (either in the same
           stream, or synchronized with add_event/wait_event operators).

    Parameters
    ----------
    mod_or_func_or_expr : Union[tvm.IRModule, tvm.relay.Function, tvm.relay.Expr]
        The given tvm IRModule, relay function or expression to verify.

    Raises
    ------
    ValueError:
        Raise this error if the given module or expression is not in ANF.
    ExecutionOrderError:
        Raise this error if there is a dependency in dataflow graph is not satisfied in the
        control graph.
    """
    if isinstance(mod_or_func_or_expr, tvm.ir.IRModule):
        expr = mod_or_func_or_expr["main"].body
    elif isinstance(mod_or_func_or_expr, tvm.relay.Function):
        expr = mod_or_func_or_expr.body
    elif isinstance(mod_or_func_or_expr, tvm.relay.Expr):
        expr = mod_or_func_or_expr
    else:
        raise ValueError(
            "Expect tvm.ir.IRModule, tvm.relay.Function, or tvm.relay.Expr, "
            f"but got {type(mod_or_func_or_expr)}."
        )

    verify_expr_schedule(expr)
