import ast
from contextlib import contextmanager
from inspect import isfunction
from numbers import Number
from typing import Callable, Dict

from tvm import relay

from .utils import OP_MAKER, NodeTransformer

SymTab = Dict[str, relay.Var]


def _convert(expr, debug):
    if debug:
        if isinstance(expr, ast.AST):
            return expr
        if isinstance(expr, Number):
            return ast.Num(n=expr)
        if isinstance(expr, str):
            return ast.Str(s=expr)
        raise NotImplementedError(expr)
    else:
        if isfunction(expr):
            return expr
        if isinstance(expr, Number):
            return lambda _: relay.const(expr, dtype="int64")
        raise NotImplementedError(expr)


class IRBuilder(object):

    def __init__(self, debug: bool):
        self.stmt_block = []
        self.scopes = []
        self.sym_names = set()
        self.debug = debug

    def get(self):
        return ast.Module(body=self.stmt_block)

    def add_sym(self, name: str) -> None:
        assert name not in self.sym_names
        self.sym_names.add(name)

    def sym_get(self, name: str) -> Callable[[SymTab], relay.Expr]:
        if self.debug:
            assert name in self.sym_names
            return ast.Name(id=name, ctx=ast.Load())
        return lambda sym_tab: sym_tab[name]


    def op(self, category: str, node_t: type, *args) -> Callable[[SymTab], relay.Expr]:
        if self.debug:
            args = [_convert(arg, debug=True) for arg in args]
            DEBUG_RULES = {
                    'unary_op': (1, lambda args: ast.UnaryOp(op=node_t(), operand=args[0])),
                    'bin_op'  : (2, lambda args: ast.BinOp(left=args[0], op=node_t(), right=args[1])),
                    'bool_op' : (2, lambda args: ast.BoolOp(op=node_t(), values=args)),
                    'compare' : (2, lambda args: ast.Compare(left=args[0], ops=[node_t()], \
                        comparators=[args[1]]))
            }
            assert category in DEBUG_RULES.keys(), "{} is not defined".format(category)
            cnt, constructor = DEBUG_RULES[category]
            assert len(args) == cnt
            return constructor(args)
        args = [_convert(arg, debug=False) for arg in args]
        maker = OP_MAKER[node_t]
        return lambda sym_tab: maker(*(arg(sym_tab) for arg in args))

    def sym_set(self, name: str, expr) -> None:
        name = ast.Name(id=name, ctx=ast.Store())
        expr = _convert(expr, self.debug)
        self.stmt_block.append(ast.Assign(targets=[name], value=expr))

    def add_pass(self) -> None:
        self.stmt_block.append(ast.Pass())

    def add_return(self, expr) -> None:
        expr = _convert(expr, self.debug)
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


def build_ir(invoker: Callable[[IRBuilder], None], debug=False) -> ast.Module:
    ib = IRBuilder(debug=debug)
    invoker(ib, ast)
    return ib.get()
