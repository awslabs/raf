import ast

from typing import List
from .utils import NodeTransformer


class CanonicalizeControlFlow(NodeTransformer):

    def __init__(self):
        super(CanonicalizeControlFlow, self).__init__(strict=False)

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


def canonicalize_control_flow(node: ast.AST) -> ast.AST:
    return CanonicalizeControlFlow().run(node)
