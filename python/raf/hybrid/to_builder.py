# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import ast
import inspect
from typing import List, Set, Tuple

from .hybrid_utils import SUPPORTED_OPS, NodeTransformer, NodeVisitor


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

    def visit_Name(self, node: ast.Name):  # pylint: disable=invalid-name
        if node.id not in self.free_names:
            self.local_names.add(node.id)

    def visit_FunctionDef(self, node: ast.FunctionDef):  # pylint: disable=invalid-name
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

    def visit_If(self, node: ast.If):  # pylint: disable=invalid-name
        return ast.If(
            test=node.test,
            body=self._canonicalize(node.body),
            orelse=self._canonicalize(node.orelse),
        )

    def visit_While(self, node: ast.While):  # pylint: disable=invalid-name
        return ast.While(test=node.test, body=self._canonicalize(node.body), orelse=[])

    def visit_FunctionDef(self, node: ast.FunctionDef):  # pylint: disable=invalid-name
        return ast.FunctionDef(
            name=node.name,
            args=node.args,
            body=self._canonicalize(node.body),
            decorator_list=node.decorator_list,
            returns=node.returns,
        )


class Unbox(NodeTransformer):
    def __init__(self):
        super(Unbox, self).__init__(strict=False)

    def run(self, node: ast.AST) -> ast.AST:
        return self.visit(node)

    def _unbox(self, lhs, rhs):
        if isinstance(lhs, ast.Name):
            assert isinstance(lhs.ctx, ast.Store)

            return [ast.Assign(targets=[lhs], value=rhs)]

        if isinstance(lhs, ast.Tuple):
            assert isinstance(lhs.ctx, ast.Store)
            ret = []

            for idx, elt in enumerate(lhs.elts):
                idx = ast.Index(value=ast.Num(idx))
                value = ast.Subscript(value=rhs, slice=idx, ctx=ast.Load())
                ret.extend(self._unbox(elt, value))

            return ret
        raise NotImplementedError()

    def _unbox_basic_block(self, stmts: List[ast.AST]):
        new_stmts = []

        for stmt in stmts:
            if isinstance(stmt, ast.Assign):
                (lhs,), rhs = stmt.targets, stmt.value
                new_stmts.extend(self._unbox(lhs, rhs))
            else:
                new_stmts.append(stmt)

        return new_stmts

    def visit_If(self, node: ast.If):  # pylint: disable=invalid-name
        return ast.If(
            test=node.test,
            body=self._unbox_basic_block(node.body),
            orelse=self._unbox_basic_block(node.orelse),
        )

    def visit_While(self, node: ast.While):  # pylint: disable=invalid-name
        return ast.While(test=node.test, body=self._unbox_basic_block(node.body), orelse=[])

    def visit_FunctionDef(self, node: ast.FunctionDef):  # pylint: disable=invalid-name
        return ast.FunctionDef(
            name=node.name,
            args=node.args,
            body=self._unbox_basic_block(node.body),
            decorator_list=node.decorator_list,
            returns=node.returns,
        )


def _call(name, *args):
    func = ast.Attribute(value=ast.Name(id="ib", ctx=ast.Load()), attr=name, ctx=ast.Load())
    call = ast.Call(func=func, args=list(args), keywords=[])

    return call


def _with(node: ast.AST, body: List[ast.AST]):
    item = ast.withitem(context_expr=node, optional_vars=None)

    return ast.With(items=[item], body=body)


def _op(category: str, node: ast.AST, *args):
    op_name = node.__class__.__name__
    category = ast.Str(s=category)
    ast_name = ast.Name(id="ast", ctx=ast.Load())
    callee = ast.Attribute(value=ast_name, attr=op_name, ctx=ast.Load())

    return _call("op", category, callee, *args)


class ToBuilder(NodeTransformer):
    def __init__(self, local_names: Set[str]):
        super(ToBuilder, self).__init__(strict=True)
        self.name = None
        self.local_names = local_names

    def run(self, node: ast.AST, name: str) -> ast.AST:
        self.name = name

        return self.visit(node)

    def default_visit(self, node: ast.AST) -> ast.AST:
        return super(ToBuilder, self).generic_visit(node)

    def visit(self, node: ast.AST) -> ast.AST:  # pylint: disable=invalid-name,arguments-differ
        if type(node) in SUPPORTED_OPS:  # pylint: disable=unidiomatic-typecheck
            return node

        return super(ToBuilder, self).visit(node)

    def visit_Name(self, node: ast.Name) -> ast.AST:  # pylint: disable=invalid-name
        assert isinstance(node.ctx, ast.Load)

        if node.id not in self.local_names:
            return node
        name = ast.Str(s=node.id)

        return _call("sym_get", name)

    def visit_Assign(self, node: ast.Assign) -> ast.AST:  # pylint: disable=invalid-name
        assert len(node.targets) == 1
        lhs, rhs = node.targets[0], node.value

        if not isinstance(lhs, ast.Name):
            raise NotImplementedError("Unsupported lhs: {}".format(lhs.__class__.__name__))
        assert isinstance(lhs.ctx, ast.Store)
        assert lhs.id in self.local_names
        name = ast.Str(s=lhs.id)
        rhs = self.visit(rhs)
        call = _call("sym_set", name, rhs)

        return ast.Expr(value=call)

    def visit_Pass(
        self, node: ast.Pass
    ):  # pylint: disable=invalid-name,no-self-use,unused-argument
        call = _call("add_pass")

        return ast.Expr(value=call)

    def visit_Return(self, node: ast.Return):  # pylint: disable=invalid-name
        value = self.visit(node.value)
        call = _call("add_return", value)

        return ast.Expr(value=call)

    def visit_Break(
        self, node: ast.Break
    ):  # pylint: disable=invalid-name,no-self-use,unused-argument
        call = _call("add_break")

        return ast.Expr(value=call)

    def visit_Continue(
        self, node: ast.Continue
    ):  # pylint: disable=invalid-name,no-self-use,unused-argument
        call = _call("add_continue")

        return ast.Expr(value=call)

    def visit_If(self, node: ast.If):  # pylint: disable=invalid-name
        test = self.visit(node.test)
        body = [self.visit(stmt) for stmt in node.body]
        orelse = [self.visit(stmt) for stmt in node.orelse]
        with_if = _with(
            node=_call("add_if", test),
            body=[_with(_call("add_then"), body=body), _with(_call("add_else"), body=orelse)],
        )

        return with_if

    def visit_While(self, node: ast.While):  # pylint: disable=invalid-name
        assert not node.orelse
        test = self.visit(node.test)
        body = [self.visit(stmt) for stmt in node.body]

        return _with(_call("add_while", test), body=body)

    def visit_FunctionDef(self, node: ast.FunctionDef):  # pylint: disable=invalid-name
        args = [ast.arg(arg="ib", annotation=None), ast.arg(arg="ast", annotation=None)]
        arguments = ast.arguments(
            args=args, vararg=None, kwonlyargs=[], kwarg=None, defaults=[], kw_defaults=[]
        )
        body = [ast.Expr(value=_call("add_sym", ast.Str(s=name))) for name in self.local_names] + [
            self.visit(stmt) for stmt in node.body
        ]

        return ast.FunctionDef(
            name=self.name, args=arguments, body=body, decorator_list=[], returns=None
        )

    def visit_UnaryOp(self, node: ast.UnaryOp):  # pylint: disable=invalid-name
        operand = self.visit(node.operand)

        return _op("unary_op", node.op, operand)

    def visit_BinOp(self, node: ast.BinOp):  # pylint: disable=invalid-name
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)

        return _op("bin_op", node.op, lhs, rhs)

    def visit_BoolOp(self, node: ast.BinOp):  # pylint: disable=invalid-name
        assert len(node.values) >= 2
        values = [self.visit(value) for value in node.values]
        result = _op("bool_op", node.op, values[0], values[1])

        for value in values[2:]:
            result = _op("bool_op", node.op, result, value)

        return result

    def visit_Compare(self, node: ast.Compare):  # pylint: disable=invalid-name
        lhs = self.visit(node.left)
        rhs = self.visit(node.comparators[0])

        return _op("compare", node.ops[0], lhs, rhs)

    def visit_Tuple(self, node: ast.Tuple):  # pylint: disable=invalid-name
        assert isinstance(node.ctx, ast.Load)
        elts = [self.visit(value) for value in node.elts]

        return _call("make_tuple", *elts)

    def visit_Subscript(self, node: ast.Subscript):  # pylint: disable=invalid-name
        assert isinstance(node.ctx, ast.Load)
        value = self.visit(node.value)
        slice_ = self.visit(node.slice)

        if isinstance(node.slice, ast.Index):
            return _call("sym_slice_index", value, slice_)
        if isinstance(node.slice, ast.Slice):
            return _call("sym_slice_strided", value, slice_.lower, slice_.upper, slice_.step)
        raise NotImplementedError

    def visit_Index(self, node: ast.Index):  # pylint: disable=invalid-name
        assert isinstance(node.value, ast.Num)
        value = self.visit(node.value)

        return value

    def visit_Slice(self, node: ast.Slice):  # pylint: disable=invalid-name
        lower = self.visit(node.lower)
        upper = self.visit(node.upper)

        if node.step:
            step = self.visit(node.step)
        else:
            step = ast.Num(n=1)

        return ast.Slice(lower=lower, upper=upper, step=step)

    # pylint: disable=pointless-statement
    ########## Module ##########
    visit_Module = default_visit
    ########## Literals ##########
    visit_Num = default_visit
    visit_Ellipsis = default_visit
    visit_NameConstant = default_visit
    ########## Variables ##########
    visit_Name
    visit_Tuple
    ########## Expressions ##########
    visit_Expr = default_visit
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
    # pylint: enable=pointless-statement


def to_builder(node: ast.AST, pyfunc, name: str) -> Tuple[ast.AST, Set[str]]:
    node = DeadCodeEliminationLite().run(node)
    local_names = LocalNames().run(node, pyfunc)
    node = Unbox().run(node)
    node = ToBuilder(local_names).run(node, name)
    ast.fix_missing_locations(node)

    return node, local_names
