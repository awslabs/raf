import ast
from tvm import relay

OP_MAKER = {
    ### UnaryOp ###
    ast.UAdd: None,  # TODO
    ast.USub: relay.negative,
    ast.Not: relay.logical_not,
    ast.Invert: None,
    ### BinOp ###
    ast.Add: relay.add,
    ast.Sub: relay.subtract,
    ast.Mult: relay.multiply,
    ast.Div: relay.divide,
    ast.FloorDiv: relay.divide,  # FIXME
    ast.Mod: relay.mod,
    ast.Pow: None,  # TODO
    ast.LShift: None,
    ast.RShift: None,
    ast.BitOr: None,
    ast.BitXor: None,
    ast.BitAnd: None,
    ast.MatMult: None,  # TODO
    ### BoolOp ###
    ast.And: relay.logical_and,
    ast.Or: relay.logical_or,
    ### Compare ###
    ast.Gt: relay.greater,
    ast.GtE: relay.greater_equal,
    ast.Lt: relay.less,
    ast.LtE: relay.less_equal,
    ast.Eq: relay.equal,
    ast.NotEq: relay.not_equal,
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


def top_type():
    # TODO(@junrushao1994): fake it until you make it
    return relay.TensorType(shape=(), dtype="int64")


def unbound_value():
    # TODO(@junrushao1994): fake it until you make it
    return relay.const(0, dtype="int64")


def get_func_name(pyfunc):
    return pyfunc.__module__ + "$" + pyfunc.__qualname__
