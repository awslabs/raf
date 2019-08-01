import ast
from .utils import NodeVisitor


class FingerTree(NodeVisitor):

    def __init__(self):
        super(FingerTree, self).__init__()

    def visit_Return(self, node: ast.Return, stmts, break_point, continue_point):
        self.finger[node] = ()
        return node

    def visit_stmts(self, stmts, break_point, continue_point):
        if stmts:
            if stmts[0] in self.finger:
                return stmts[0]
            return self.visit(stmts[0], stmts[1:], break_point, continue_point)
        elif continue_point:
            return continue_point

    def visit_While(self, node: ast.While, stmts, break_point, continue_point):
        self.finger[node] = None
        left = self.visit_stmts(node.body, stmts[0], node)
        right = self.visit_stmts(stmts, break_point, continue_point)
        self.finger[node] = (left, right)
        return node

    def visit_If(self, node: ast.If, stmts, break_point, continue_point):
        self.finger[node] = None
        left = self.visit_stmts(node.body + stmts, break_point, continue_point)
        right = self.visit_stmts(
            node.orelse + stmts, break_point, continue_point)
        self.finger[node] = (left, right)
        return node

    def visit_Module(self, node: ast.Module, stmts, break_point, continue_point):
        return self.visit_stmts(node.body, break_point, continue_point)

    def visit_Assign(self, node: ast.Assign, stmts, break_point, continue_point):
        self.finger[node] = []
        value = self.visit_stmts(stmts, break_point, continue_point)
        if value:
            self.finger[node] = (value, )
        return node

    def visit_Break(self, node: ast.Break, stmts, break_point, continue_point):
        return break_point

    def visit_Continue(self, node: ast.Continue, stmts, break_point, continue_point):
        return continue_point

    def visit_Pass(self, node: ast.Pass, stmts, break_point, continue_point):
        self.finger[node] = None
        value = self.visit_stmts(stmts, break_point, continue_point)
        if value:
            self.finger[node] = (value, )
        return node

    def run(self, node: ast.AST):
        self.finger = {}
        entry = self.visit(node, None, None, None)
        return entry, self.finger
