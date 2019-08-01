import ast
import numbers
import numpy as np
from contextlib import contextmanager

from tvm import relay
from .utils import OP_MAKER


class DebugIRBuilder(object):

    def __init__(self):
        self.stmt_block = []
        self.scopes = []

    def get(self):
        return ast.Module(body=self.stmt_block)

    @staticmethod
    def _to_ast(arg):
        if isinstance(arg, ast.AST):
            return arg
        if isinstance(arg, numbers.Number):
            return ast.Num(n=arg)
        if isinstance(arg, str):
            return ast.Str(s=arg)
        raise NotImplementedError(arg)

    def add_sym(self, name: str, type_: relay.Type = None) -> None:
        pass

    def sym_get(self, name: str):
        return ast.Name(id=name, ctx=ast.Load())

    def sym_set(self, name: str, expr):
        expr = DebugIRBuilder._to_ast(expr)
        name = ast.Name(id=name, ctx=ast.Store())
        stmt = ast.Assign(targets=[name], value=expr)
        self.stmt_block.append(stmt)

    def op(self, category: str, node_t, *args):
        args = [DebugIRBuilder._to_ast(arg) for arg in args]
        if category == "unary_op":
            assert len(args) == 1
            result = ast.UnaryOp(op=node_t(), operand=args[0])
        elif category == "bin_op":
            assert len(args) == 2
            result = ast.BinOp(left=args[0], op=node_t(), right=args[1])
        elif category == "bool_op":
            assert len(args) == 2
            result = ast.BoolOp(op=node_t(), values=args)
        elif category == "compare":
            assert len(args) == 2
            result = ast.Compare(left=args[0], ops=[
                                 node_t()], comparators=[args[1]])
        else:
            raise AssertionError("{} is not defined".format(category))
        return result

    def add_pass(self) -> None:
        self.stmt_block.append(ast.Pass())

    def add_return(self, expr) -> None:
        self.stmt_block.append(ast.Return(value=expr))

    def add_break(self) -> None:
        self.stmt_block.append(ast.Break())

    def add_continue(self) -> None:
        self.stmt_block.append(ast.Continue())

    @contextmanager
    def add_if(self, expr):
        try:
            self.scopes.append((list(), list()))
            yield
        finally:
            then, else_ = self.scopes.pop()
            self.stmt_block.append(ast.If(test=expr, body=then, orelse=else_))

    @contextmanager
    def add_then(self):
        try:
            prev_stmt = self.stmt_block
            self.stmt_block, _ = self.scopes[-1]
            yield
        finally:
            self.stmt_block = prev_stmt

    @contextmanager
    def add_else(self):
        try:
            prev_stmt = self.stmt_block
            _, self.stmt_block = self.scopes[-1]
            yield
        finally:
            self.stmt_block = prev_stmt

    @contextmanager
    def add_while(self, expr):
        try:
            self.scopes.append(list())
            prev_stmt = self.stmt_block
            self.stmt_block = self.scopes[-1]
            yield
        finally:
            body = self.scopes.pop()
            self.stmt_block = prev_stmt
            self.stmt_block.append(ast.While(test=expr, body=body, orelse=[]))
