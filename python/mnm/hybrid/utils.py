import ast

from .._core.op import get_op
from .._core.ir import ConstantExpr
from .._core.value import IntValue


def _wrap_op(name):
    op = get_op(name)
    return lambda *args: op(eager=False, args=args, attrs=None)


OP_MAKER = {
    ### UnaryOp ###
    ast.UAdd: None,
    ast.USub: _wrap_op("mnm.op.negative"),
    ast.Not: None,
    ast.Invert: None,
    ### BinOp ###
    ast.Add: _wrap_op("mnm.op.add"),
    ast.Sub: _wrap_op("mnm.op.subtract"),
    ast.Mult: _wrap_op("mnm.op.multiply"),
    ast.Div: _wrap_op("mnm.op.divide"),
    ast.FloorDiv: _wrap_op("mnm.op.divide"),  # FIXME
    ast.Mod: _wrap_op("mnm.op.mod"),
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
    ast.Gt: _wrap_op("mnm.op.greater"),
    ast.GtE: _wrap_op("mnm.op.greater_equal"),
    ast.Lt: _wrap_op("mnm.op.less"),
    ast.LtE: _wrap_op("mnm.op.less_equal"),
    ast.Eq: _wrap_op("mnm.op.equal"),
    ast.NotEq: _wrap_op("mnm.op.not_equal"),
    ast.Is: None,
    ast.IsNot: None,
    ast.In: None,
    ast.NotIn: None,
}

SUPPORTED_OPS = set(py_op for py_op, relay_op in OP_MAKER.items() if relay_op)


class NodeVisitor(object):

    def __init__(self, strict=True):
        self.strict = strict

    def visit(self, node, *args, **kwargs):
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, None)
        if visitor is None:
            if not self.strict:
                return self.generic_visit(node, *args, **kwargs)
            raise NotImplementedError("{} is not supported in {}".format(
                node.__class__.__name__, self.__class__.__name__))
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
                        elif not isinstance(value, ast.AST):
                            new_values.extend(value)
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
    return ConstantExpr(IntValue(0))


def get_func_name(pyfunc):
    return pyfunc.__module__ + "$" + pyfunc.__qualname__
