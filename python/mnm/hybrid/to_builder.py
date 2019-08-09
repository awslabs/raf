import ast
import inspect
from typing import List, Set, Tuple

from .utils import SUPPORTED_OPS, NodeTransformer, NodeVisitor


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

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.visit(node.args)
        for stmt in node.body:
            self.visit(stmt)


class DeadCodeEliminationLite(NodeTransformer):

    def __init__(self):
        super(DeadCodeEliminationLite, self).__init__(strict=False)

    def _canonicalize(self, stmts: List[ast.AST]):
        new_stmts = []
        for stmt in stmts:
            if isinstance(stmt, (ast.Pass, ast.Ellipsis)):
                continue
            if isinstance(stmt, (ast.Break, ast.Continue, ast.Return)):
                new_stmts.append(stmt)
                break
            new_stmts.append(self.visit(stmt))
        else:
            new_stmts.append(ast.Pass())
        assert len(new_stmts) > 0
        return new_stmts

    def run(self, node: ast.AST) -> ast.AST:
        return self.visit(node)

    def visit_If(self, node: ast.If):
        return ast.If(test=node.test,
                      body=self._canonicalize(node.body),
                      orelse=self._canonicalize(node.orelse))

    def visit_While(self, node: ast.While):
        return ast.While(test=node.test,
                         body=self._canonicalize(node.body),
                         orelse=[])

    def visit_FunctionDef(self, node: ast.FunctionDef):
        return ast.FunctionDef(name=node.name,
                               args=node.args,
                               body=self._canonicalize(node.body),
                               decorator_list=node.decorator_list,
                               returns=node.returns)


def _call(name, *args):
    func = ast.Attribute(value=ast.Name(
        id='ib', ctx=ast.Load()), attr=name, ctx=ast.Load())
    call = ast.Call(func=func, args=list(args), keywords=[])
    return call


def _with(node: ast.AST, body: List[ast.AST]):
    item = ast.withitem(context_expr=node, optional_vars=None)
    return ast.With(items=[item], body=body)


def _op(category: str, node: ast.AST, *args):
    op_name = node.__class__.__name__
    category = ast.Str(s=category)
    ast_name = ast.Name(id="ast", ctx=ast.Load())
    op = ast.Attribute(ast_name, op_name, ctx=ast.Load())
    return _call("op", category, op, *args)


class ToBuilder(NodeTransformer):

    def __init__(self, local_names: Set[str]):
        super(ToBuilder, self).__init__(strict=True)
        self.local_names = local_names

    def run(self, node: ast.AST, name: str) -> ast.AST:
        self.name = name
        return self.visit(node)

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
        return _call("sym_get", name)

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
        call = _call("sym_set", name, rhs)
        return ast.Expr(value=call)

    def visit_Pass(self, node: ast.Pass):
        call = _call("add_pass")
        return ast.Expr(value=call)

    def visit_Return(self, node: ast.Return):
        value = self.visit(node.value)
        call = _call("add_return", value)
        return ast.Expr(value=call)

    def visit_Break(self, node: ast.Break):
        call = _call("add_break")
        return ast.Expr(value=call)

    def visit_Continue(self, node: ast.Continue):
        call = _call("add_continue")
        return ast.Expr(value=call)

    def visit_If(self, node: ast.If):
        test = self.visit(node.test)
        body = [self.visit(stmt) for stmt in node.body]
        orelse = [self.visit(stmt) for stmt in node.orelse]
        with_if = _with(
            node=_call("add_if", test),
            body=[_with(_call("add_then"), body=body),
                  _with(_call("add_else"), body=orelse)])
        return with_if

    def visit_While(self, node: ast.While):
        assert not node.orelse
        test = self.visit(node.test)
        body = [self.visit(stmt) for stmt in node.body]
        return _with(_call("add_while", test), body=body)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        args = [ast.arg(arg="ib", annotation=None),
                ast.arg(arg="ast", annotation=None)]
        arguments = ast.arguments(
            args=args, vararg=None, kwonlyargs=[], kwarg=None, defaults=[], kw_defaults=[])
        body = [ast.Expr(value=_call("add_sym", ast.Str(s=name)))
                for name in self.local_names] + [self.visit(stmt) for stmt in node.body]
        return ast.FunctionDef(name=self.name, args=arguments, body=body, decorator_list=[], returns=None)

    def visit_UnaryOp(self, node: ast.UnaryOp):
        operand = self.visit(node.operand)
        return _op("unary_op", node.op, operand)

    def visit_BinOp(self, node: ast.BinOp):
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        return _op("bin_op", node.op, lhs, rhs)

    def visit_BoolOp(self, node: ast.BinOp):
        assert len(node.values) >= 2
        values = [self.visit(value) for value in node.values]
        result = _op("bool_op", node.op, values[0], values[1])
        for value in values[2:]:
            result = _op("bool_op", node.op, result, value)
        return result

    def visit_Compare(self, node: ast.Compare):
        lhs = self.visit(node.left)
        rhs = self.visit(node.comparators[0])
        return _op("compare", node.ops[0], lhs, rhs)

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


def to_builder(node: ast.AST, pyfunc, name: str) -> Tuple[ast.AST, Set[str]]:
    node = DeadCodeEliminationLite().run(node)
    local_names = LocalNames().run(node, pyfunc)
    node = ToBuilder(local_names).run(node, name)
    ast.fix_missing_locations(node)
    return node, local_names
