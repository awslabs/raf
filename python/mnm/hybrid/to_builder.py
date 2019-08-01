import ast
import inspect

from typing import Set, List
from .utils import NodeVisitor, NodeTransformer, SUPPORTED_OPS


class LocalNames(NodeVisitor):

    def __init__(self):
        super(LocalNames, self).__init__(strict=False)
        self.free_names = set()
        self.local_names = set()

    def run(self, node: ast.AST, pyfunc):
        closure_vars = inspect.getclosurevars(pyfunc)
        self.free_names |= closure_vars.nonlocals.keys()
        self.free_names |= closure_vars.globals.keys()
        self.free_names |= closure_vars.builtins.keys()
        self.free_names |= closure_vars.unbound
        self.visit(node)
        return self.local_names

    def visit_arg(self, node: ast.arg):
        if node.arg not in self.free_names:
            self.local_names.add(node.arg)

    def visit_Name(self, node: ast.Name):
        if node.id not in self.free_names:
            self.local_names.add(node.id)


class ToBuilder(NodeTransformer):

    def __init__(self, local_names: Set[str]):
        super(ToBuilder, self).__init__(strict=True)
        self.local_names = local_names

    def run(self, node: ast.AST) -> ast.AST:
        return self.visit(node)

    @staticmethod
    def _call(name, *args):
        func = ast.Attribute(value=ast.Name(
            id='ib', ctx=ast.Load()), attr=name, ctx=ast.Load())
        call = ast.Call(func=func, args=args, keywords=[])
        return call

    @staticmethod
    def _with(node: ast.AST, body: List[ast.AST]):
        item = ast.withitem(context_expr=node, optional_vars=None)
        return ast.With(items=[item], body=body)

    @staticmethod
    def _op(category: str, node: ast.AST, *args):
        op_name = node.__class__.__name__
        category = ast.Str(s=category)
        ast_name = ast.Name(id="ast", ctx=ast.Load())
        op = ast.Attribute(ast_name, op_name, ctx=ast.Load())
        return ToBuilder._call("op", category, op, *args)

    def generic_visit(self, node: ast.AST) -> ast.AST:
        return super(ToBuilder, self).generic_visit(node)

    def visit(self, node: ast.AST) -> ast.AST:
        if type(node) in SUPPORTED_OPS:
            return node
        return super(ToBuilder, self).visit(node)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        assert isinstance(node.ctx, ast.Load)
        if node.id not in self.local_names:
            return node
        name = ast.Str(s=node.id)
        return ToBuilder._call("sym_get", name)

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        assert len(node.targets) == 1
        lhs, rhs = node.targets[0], node.value
        if not isinstance(lhs, ast.Name):
            raise NotImplementedError(
                "Unsupported lhs: {}".format(lhs.__class__.__name__))
        assert isinstance(lhs.ctx, ast.Store)
        assert lhs.id in self.local_names
        name = ast.Str(s=lhs.id)
        rhs = self.visit(rhs)
        call = ToBuilder._call("sym_set", name, rhs)
        return ast.Expr(value=call)

    def visit_Pass(self, node: ast.Pass):
        call = ToBuilder._call("add_pass")
        return ast.Expr(value=call)

    def visit_Return(self, node: ast.Return):
        value = self.visit(node.value)
        call = ToBuilder._call("add_return", value)
        return ast.Expr(value=call)

    def visit_Break(self, node: ast.Break):
        call = ToBuilder._call("add_break")
        return ast.Expr(value=call)

    def visit_Continue(self, node: ast.Continue):
        call = ToBuilder._call("add_continue")
        return ast.Expr(value=call)

    def visit_If(self, node: ast.If):
        test = self.visit(node.test)
        body = [self.visit(stmt) for stmt in node.body]
        orelse = [self.visit(stmt) for stmt in node.orelse]
        with_if = ToBuilder._with(
            node=ToBuilder._call("add_if", test),
            body=[ToBuilder._with(ToBuilder._call("add_then"), body=body),
                  ToBuilder._with(ToBuilder._call("add_else"), body=orelse)])
        return with_if

    def visit_While(self, node: ast.While):
        assert not node.orelse
        test = self.visit(node.test)
        body = [self.visit(stmt) for stmt in node.body]
        return ToBuilder._with(ToBuilder._call("add_while", test), body=body)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        name = "__build_" + node.name
        args = [ast.arg(arg="ib", annotation=None)]
        arguments = ast.arguments(
            args=args, vararg=None, kwonlyargs=None, kwarg=None, defaults=[], kw_defaults=None)
        body = [self.visit(stmt) for stmt in node.body]
        return ast.FunctionDef(name=name, args=arguments, body=body, decorator_list=[], returns=None)

    def visit_UnaryOp(self, node: ast.UnaryOp):
        operand = self.visit(node.operand)
        return ToBuilder._op("unary_op", node.op, operand)

    def visit_BinOp(self, node: ast.BinOp):
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        return ToBuilder._op("bin_op", node.op, lhs, rhs)

    def visit_BoolOp(self, node: ast.BinOp):
        assert len(node.values) >= 2
        values = [self.visit(value) for value in node.values]
        result = ToBuilder._op("bool_op", node.op, values[0], values[1])
        for value in values[2:]:
            result = ToBuilder._op("bool_op", node.op, result, value)
        return result

    def visit_Compare(self, node: ast.Compare):
        lhs = self.visit(node.left)
        rhs = self.visit(node.comparators[0])
        return ToBuilder._op("compare", node.ops[0], lhs, rhs)

    ########## Module ##########
    visit_Module = generic_visit
    ########## Literals ##########
    visit_Num = generic_visit
    visit_Ellipsis = generic_visit
    visit_NameConstant = generic_visit
    ########## Variables ##########
    visit_Name
    ########## Expressions ##########
    visit_Expr = generic_visit
    visit_UnaryOp
    visit_BinOp
    visit_BoolOp
    visit_Compare
    ########## Statements ##########
    visit_Assign
    visit_Pass
    visit_If
    visit_While
    visit_Break
    visit_Continue
    visit_Return
    ########## Function and class definitions ##########
    visit_FunctionDef


def to_builder(node: ast.AST, pyfunc) -> ast.AST:
    local_names = LocalNames().run(node, pyfunc)
    node = ToBuilder(local_names).run(node)
    return node
