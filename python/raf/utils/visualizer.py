# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
The visualizer utilities.
"""
# pylint: disable=too-many-arguments, too-many-instance-attributes, missing-function-docstring
# pylint: disable=too-many-public-methods
from collections import defaultdict
import os
import pydot
import tvm
from tvm import relay
from raf._core.value import FloatValue, IntValue, StringValue
from raf._ffi.ir.constant import ExtractValue


class DataflowGraphDrawer(tvm.relay.ExprFunctor):
    """
    The dataflow graph drawer.
    """

    def __init__(
        self,
        expr,
        always_draw_exprs=None,
        graph_label="",
        draw_atomic_nodes=False,
        draw_control_nodes=False,
    ):
        """
        Init the dataflow graph drawer

        Parameters
        ----------
        expr : tvm.relay.Expr
            The relay expression to draw.
        always_draw_exprs : List[tvm.relay.Expr]
            The set of relay expressions that we should always draw.
        graph_label : str
            The label of the graph, which would be drawn at the bottom of the image.
        draw_atomic_nodes : bool
            Whether to draw atomic nodes. All nodes except Call, Tuple, and TupleGetItem are atomic
            nodes.
        draw_control_nodes : bool
            Whether to draw control nodes, including event nodes and barrier nodes.
        """
        super().__init__()
        # the expr to draw
        self.expr = expr
        # the set of exprs that always should be drawn
        self.always_draw_exprs = always_draw_exprs if always_draw_exprs else []
        # whether to draw atomic nodes
        self.draw_atomic_nodes = draw_atomic_nodes
        # whether to draw control nodes
        self.draw_control_nodes = draw_control_nodes
        # the pydot graph object
        self.graph = pydot.Dot(graph_type="digraph", label=graph_label)

        # stream-related members:
        # mapping from event id to the expr captured by the event
        self.event_expr = {}
        # the current stream id
        self.current_stream = 0
        # the communication stream id
        self.communication_stream = None
        # mapping from stream id to the launched exprs
        self.stream_exprs = defaultdict(list)
        # mapping from stream id to the events to wait for the next expr on that stream
        self.stream_wait_events = defaultdict(list)
        # mapping from stream id to the barrier to wait for the next expr on that stream
        self.stream_barrier = defaultdict(type(None))
        # the previous global barrier, used to synchronize with new stream
        self.previous_barrier = None

    def draw(self) -> pydot.Dot:
        """
        Draw the dataflow graph and saves it as a pydot Graph, which can be saved to image.

        Returns
        -------
        graph : pydot.Dot
            A pydot graph.
        """
        self.visit(self.expr)
        return self.graph

    def need_draw(self, e):
        """
        Check whether to need to draw expression e.
        Parameters
        ----------
        e : tvm.relay.Expr
            The expression to check.
        Returns
        -------
        ret : bool
            Whether to draw.
        """
        # Always draw some exprs
        if e in self.always_draw_exprs:
            return True
        # Do not draw atomic expr if draw_atomic_nodes is turned off
        if self.is_atomic_expr(e) and not self.draw_atomic_nodes:
            return False
        # Draw all non-atomic expr
        return True

    @staticmethod
    def is_atomic_expr(e):
        """
        Check the given expression is an atomic expression.
        Parameters
        ----------
        e : tvm.relay.Expr
            The given expression.

        Returns
        -------
        ret : bool
            Whether the given expression is an atomic expression.
        """
        return not isinstance(e, (relay.Call, relay.Tuple, relay.TupleGetItem))

    @staticmethod
    def is_scalar_value(value):
        """
        Check the given raf value is a scalar value (i.e., IntValue, FloatValue, StringValue).
        Parameters
        ----------
        value : raf.value.Value
            The given value.

        Returns
        -------
        ret : bool
            Whether the given value is a scalar value.
        """
        return isinstance(value, (IntValue, FloatValue, StringValue))

    @staticmethod
    def get_fused_op_name(func):
        """
        Get the name of a fused function. The name contains the names of all sub-functions and
        starts with 'fused_'.

        Parameters
        ----------
        func : tvm.relay.Function
            The fused function.

        Returns
        -------
        ret : str
            The name of the fused function.
        """

        class FusedOpCalleeVisitor(tvm.relay.ExprVisitor):
            """
            Collect the callee names in the relay function.
            """

            def __init__(self, func):
                super().__init__()
                self.func = func
                self.callees = []

            def get_callee_names(self):
                self.visit(func)
                return self.callees

            def visit_call(self, call):
                op = call.op
                if isinstance(op, tvm.ir.Op):
                    self.callees.append(op.name.split(".")[-1])
                elif isinstance(op, tvm.relay.Function):
                    self.callees.extend(FusedOpCalleeVisitor(op).get_callee_names())
                tvm.relay.ExprVisitor.visit_call(self, call)

        callee_names = FusedOpCalleeVisitor(func).get_callee_names()
        return "fused_" + "_".join(callee_names)

    def add_node(self, e, label, control_node=False, stream=None):
        """
        Add a node to the pydot graph, and add it to the operators of the specified stream.

        Parameters
        ----------
        e : tvm.relay.Expr
            The expression (node) to add.
        label : str
            The label for the node.
        control_node : bool
            Whether the node is a control node.
        stream : int | None
            Add the node to the specified stream. None for the current stream (default).
        """
        if self.need_draw(e):
            node_style = {"shape": "box", "style": '"rounded,filled"'}
            colors = [
                "beige",
                "azure",
                "burlywood",
                "coral1",
                "darkgoldenrod",
                "darkgreen",
                "darkorchid2",
                "firebrick1",
                "gold1",
                "antiquewhite",
                "aquamarine",
                "chartreuse1",
                "crimson",
            ]
            if stream is None:
                stream = self.current_stream
            node_style["fillcolor"] = colors[stream % len(colors)]
            if e in self.always_draw_exprs:
                node_style["fillcolor"] = "white"
            if control_node:
                node_style["shape"] = "diamond"
                node_style["fillcolor"] = "white"
            node = pydot.Node(name=str(len(self.memo_map)), label=label, **node_style)
            self.stream_exprs[stream].append(e)
            self.graph.add_node(node)
            self.memo_map[e] = node
        else:
            self.memo_map[e] = None

    def wait_events_and_barrier(self, e, stream=None):
        """
        Add control edges from the nodes that the given node's stream waits for to the given node.

        Parameters
        ----------
        e : tvm.relay.Expr
            The relay expression (node).
        stream : int | None
            The stream where the given node is executed on. None for current stream (default).
        """
        if stream is None:
            stream = self.current_stream
        # add the control edge derived from add_event/wait_event
        event_ids = self.stream_wait_events[stream]
        for event_id in event_ids:
            self.add_edge(self.event_expr[event_id], e, control_edge=True)
        self.stream_wait_events[stream].clear()

        # add the control edge derived from stream_barrier
        if self.stream_barrier[stream]:
            self.add_edge(self.stream_barrier[stream], e, control_edge=True)
            self.stream_barrier[stream] = None

    def add_edge(self, u_expr, v_expr, control_edge=False):
        """
        Add an edge from a node to another node to the pydot graph.

        Parameters
        ----------
        u_expr : tvm.relay.Expr
            The source node.
        v_expr : tvm.relay.Expr
            The dest node.
        control_edge : bool
            Whether this edge is a control edge. Control edges would be drawn in dashed lines.
        """
        u_node = self.memo_map[u_expr]
        v_node = self.memo_map[v_expr]
        if u_node and v_node:
            edge_style = {}
            if control_edge:
                edge_style["style"] = "dashed"
            self.graph.add_edge(pydot.Edge(u_node, v_node, **edge_style))

    def visit_function(self, e):
        attrs = e.attrs
        if attrs and "Primitive" in attrs and attrs["Primitive"] == 1:
            self.add_node(e, self.get_fused_op_name(e))
            return self.memo_map[e]
        raise NotImplementedError("Does not support a graph with non-primitive functions")

    def visit_let(self, let):
        self.visit(let.value)
        self.memo_map[let.var] = self.memo_map[let.value]
        self.visit(let.body)
        self.memo_map[let] = None
        return self.memo_map[let]

    def visit_call(self, call):
        # pylint: disable=too-many-nested-blocks, too-many-branches, too-many-statements
        op = call.op

        # deal with the schedule-related op specially
        schedule_ops = [
            "raf.op.set_stream",
            "raf.op.add_event",
            "raf.op.wait_event",
            "raf.op.stream_barrier",
        ]
        if isinstance(op, tvm.ir.Op) and op.name in schedule_ops:
            if op.name == "raf.op.set_stream":
                self.current_stream = ExtractValue(call.args[1]).value
                if self.current_stream not in self.stream_exprs and self.previous_barrier:
                    self.stream_barrier[self.current_stream] = self.previous_barrier
                self.memo_map[call] = None
            else:
                if self.draw_control_nodes:
                    if op.name == "raf.op.stream_barrier":
                        self.add_node(call, "Barrier", True)
                        self.stream_exprs[self.current_stream].pop()
                        for stream_id in self.stream_exprs:
                            if len(self.stream_exprs[stream_id]) > 0:
                                self.add_edge(
                                    self.stream_exprs[stream_id][-1], call, control_edge=True
                                )
                            self.stream_barrier[stream_id] = call
                        self.previous_barrier = call
                    else:
                        event_id = ExtractValue(call.args[0]).value
                        stream_id = ExtractValue(call.args[1]).value
                        if stream_id == -1:
                            stream_id = self.current_stream
                        if op.name == "raf.op.add_event":
                            self.add_node(
                                call,
                                f"Event({event_id}, {stream_id})",
                                control_node=True,
                                stream=stream_id,
                            )
                            if len(self.stream_exprs[stream_id]) > 1:
                                prev_expr = self.stream_exprs[stream_id][-2]
                                self.add_edge(prev_expr, call, control_edge=True)
                            self.wait_events_and_barrier(call, stream=stream_id)
                            self.event_expr[event_id] = call
                        elif op.name == "raf.op.wait_event":
                            self.stream_wait_events[stream_id].append(event_id)
                            self.memo_map[call] = None
        else:
            self.visit(op)
            stream = self.current_stream
            if isinstance(op, tvm.ir.Op):
                if op.get_attr("TRAFCollective"):
                    if self.communication_stream is None:
                        # when data parallel is enabled, we currently assume all communication ops
                        # are executed in a single stream, while all computation ops are executed
                        # in another. when we encounter a collective op, we should have at least
                        # encountered a computation op and a wait_event op
                        all_stream_ids = (
                            set(self.stream_exprs.keys())
                            .union(self.stream_wait_events.keys())
                            .union(self.stream_barrier.keys())
                        )
                        if not (len(all_stream_ids) == 2 and len(self.stream_exprs) == 1):
                            # the two-stream assumption does not hold, or the scheduling
                            # ops are not used properly
                            print(
                                "[WARNING] Cannot find valid communication stream. \
                                  Drawing communication ops on the default stream."
                            )
                            self.communication_stream = 0
                        else:
                            self.communication_stream = list(
                                all_stream_ids - set(self.stream_exprs.keys())
                            )[0]
                    stream = self.communication_stream
                    self.add_node(call, f'Call({op.name.split(".")[-1]})', stream=stream)
                else:
                    self.add_node(call, f'Call({op.name.split(".")[-1]})')
            else:
                self.add_node(call, f"Call({self.get_fused_op_name(op)})")
            self.wait_events_and_barrier(call, stream=stream)
            self.add_edge(op, call)
            for arg in call.args:
                self.visit(arg)
                self.add_edge(arg, call)
        return self.memo_map[call]

    def visit_var(self, var):
        last_name = var.name_hint.split(".")[-1]
        self.add_node(var, f"Var({last_name})")
        return self.memo_map[var]

    def visit_tuple(self, tup):
        self.add_node(tup, "Tuple")
        self.wait_events_and_barrier(tup)
        for field in tup.fields:
            self.visit(field)
            self.add_edge(field, tup)
        return self.memo_map[tup]

    def visit_tuple_getitem(self, tup_item):
        self.add_node(tup_item, f"TupleGetItem({tup_item.index})")
        self.wait_events_and_barrier(tup_item)
        self.visit(tup_item.tuple_value)
        self.add_edge(tup_item.tuple_value, tup_item)
        return self.memo_map[tup_item]

    def visit_global_var(self, global_var):
        assert isinstance(global_var, tvm.ir.GlobalVar)
        self.add_node(global_var, f"GlobalVar({global_var.name_hint})")
        return self.memo_map[global_var]

    def visit_op(self, op):
        last_name = op.name.split(".")[-1]
        self.add_node(op, f"Op({last_name})")
        return self.memo_map[op]

    def visit_constant(self, const):
        value = ExtractValue(const)
        if self.is_scalar_value(value):
            label = f"Scalar({str(value)})"
        else:
            label = "Constant"
        self.add_node(const, label)
        return self.memo_map[const]

    def visit_type(self, typ):
        return typ

    def visit_ref_create(self, _):
        raise NotImplementedError()

    def visit_ref_write(self, _):
        raise NotImplementedError()

    def visit_ref_read(self, _):
        raise NotImplementedError()

    def visit_constructor(self, _):
        raise NotImplementedError()

    def visit_match(self, _):
        raise NotImplementedError()

    def visit_if(self, _):
        raise NotImplementedError()


def draw_dataflow_graph(
    mod_or_func_or_expr,
    out_file_name="./graph.png",
    graph_label="Dataflow Graph",
    num_inputs=1,
    draw_atomic_nodes=False,
    draw_control_nodes=False,
):
    """
    Draw the dataflow graph of given module, relay function or expression. When a module is given,
    the 'main' function is drawn. The input expr or function can be either GNF, BBNF, or ANF. If
    the given function or expression are scheduled (i.e. after StreamSchedule pass), nodes on
    different CUDA streams would be in different color.

    Parameters
    ----------
    mod_or_func_or_expr : Union[tvm.ir.IRModule, tvm.relay.Function, tvm.ir.RelayExpr]
        The ir module, relay function or expression to be drawn. If a module is given, the main
        function is drawn. We can use draw_dataflow_graph(mod['other_func']) to draw other function
        in the ir module (in this example, 'other_func' is the function's name), if needed.

    out_file_name : str
        The output file name to save the image. Default: './graph.png'.

    graph_label : str
        The graph label to be shown in the image. Default: 'Dataflow Graph'.

    num_inputs : int
        When drawing a function, the first num_inputs of the parameters are always drawn, no matter
        whether draw_atomic_nodes is turned on or off. Default: 1.

    draw_atomic_nodes : bool
        Whether to draw the atomic nodes. We take all expressions other than Call, Tuple and
        TupleGetItem as atomic nodes. Default: False.

    draw_control_nodes : bool
        Whether to draw the control node and control dependency. All data dependency are drawn in
        solid line and the control dependency are drawn in dashed line. Default: False.
    """
    if isinstance(mod_or_func_or_expr, tvm.ir.IRModule):
        expr = mod_or_func_or_expr["main"].body
        always_draw_exprs = mod_or_func_or_expr["main"].params[:num_inputs]
    elif isinstance(mod_or_func_or_expr, tvm.relay.Function):
        expr = mod_or_func_or_expr.body
        always_draw_exprs = mod_or_func_or_expr.params[:num_inputs]
    elif isinstance(mod_or_func_or_expr, tvm.relay.Expr):
        expr = mod_or_func_or_expr
        always_draw_exprs = []
    else:
        raise ValueError(
            "Expect tvm.ir.IRModule, tvm.relay.Function, or tvm.relay.Expr, "
            f"but {type(mod_or_func_or_expr)} got."
        )

    drawer = DataflowGraphDrawer(
        expr,
        always_draw_exprs=always_draw_exprs,
        graph_label=graph_label,
        draw_atomic_nodes=draw_atomic_nodes,
        draw_control_nodes=draw_control_nodes,
    )
    dgraph = drawer.draw()

    dirname = os.path.dirname(out_file_name)
    if dirname != "":
        os.makedirs(dirname, exist_ok=True)
    dgraph.write(out_file_name, format="png")
