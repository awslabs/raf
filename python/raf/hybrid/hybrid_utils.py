# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import ast

from raf._core.value import IntValue, Value
from raf._lib import _get_global_func, relay

_GET_OP = _get_global_func("ir.GetOp")


def _wrap_op(name):
    return lambda *args: relay.Call(op=_GET_OP(name), args=args, attrs=None)


OP_MAKER = {
    ### UnaryOp ###
    ast.UAdd: None,
    ast.USub: _wrap_op("raf.op.negative"),
    ast.Not: None,
    ast.Invert: None,
    ### BinOp ###
    ast.Add: _wrap_op("raf.op.add"),
    ast.Sub: _wrap_op("raf.op.subtract"),
    ast.Mult: _wrap_op("raf.op.multiply"),
    ast.Div: _wrap_op("raf.op.divide"),
    ast.FloorDiv: _wrap_op("raf.op.divide"),  # FIXME
    ast.Mod: _wrap_op("raf.op.mod"),
    ast.Pow: None,
    ast.LShift: None,
    ast.RShift: None,
    ast.BitOr: None,
    ast.BitXor: None,
    ast.BitAnd: None,
    ast.MatMult: None,
    ### BoolOp ###
    ast.And: None,
    ast.Or: None,
    ### Compare ###
    ast.Gt: _wrap_op("raf.op.greater"),
    ast.GtE: _wrap_op("raf.op.greater_equal"),
    ast.Lt: _wrap_op("raf.op.less"),
    ast.LtE: _wrap_op("raf.op.less_equal"),
    ast.Eq: _wrap_op("raf.op.equal"),
    ast.NotEq: _wrap_op("raf.op.not_equal"),
    ast.Is: None,
    ast.IsNot: None,
    ast.In: None,
    ast.NotIn: None,
}

SUPPORTED_OPS = set(py_op for py_op, relay_op in OP_MAKER.items() if relay_op)


class NodeVisitor:
    def __init__(self, strict=True):
        self.strict = strict

    def visit(self, node, *args, **kwargs):
        method = "visit_" + node.__class__.__name__
        visitor = getattr(self, method, None)

        if visitor is None:
            if not self.strict:
                return self.generic_visit(node, *args, **kwargs)
            raise NotImplementedError(
                "{} is not supported in {}".format(node.__class__.__name__, self.__class__.__name__)
            )

        return visitor(node, *args, **kwargs)

    def generic_visit(self, node, *args, **kwargs):
        # XXX: It doesn't deal with return value.

        for _, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        self.visit(item, *args, **kwargs)
            elif isinstance(value, ast.AST):
                self.visit(value, *args, **kwargs)


class NodeTransformer(NodeVisitor):
    def generic_visit(self, node, *args, **kwargs):
        for field, old_value in ast.iter_fields(node):
            if isinstance(old_value, list):
                new_values = []

                for value in old_value:
                    if isinstance(value, ast.AST):
                        value = self.visit(value, *args, **kwargs)

                        if value is None:
                            continue
                    new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, ast.AST):
                new_node = self.visit(old_value, *args, **kwargs)

                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)

        return node


def unbound_constant_expr():
    # TODO(@junrushao1994): fake it until you make it

    return Value.as_const_expr(IntValue(0))
