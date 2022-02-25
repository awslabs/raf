# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import ast
from collections import defaultdict
from typing import Dict, List, Tuple, Union

from .hybrid_utils import NodeVisitor, unbound_constant_expr

# Tuple[()] : Return
# Tuple[ast.AST]: unconditional jump
# Tuple[ast.AST, ast.AST]: true, false
Finger = Union[Tuple[()], Tuple[ast.AST], Tuple[ast.AST, ast.AST]]


class FingerFinder(NodeVisitor):
    def __init__(self):
        super(FingerFinder, self).__init__()
        self.finger = None

    def visit_stmts(self, stmts, break_point, continue_point):
        if stmts:
            if stmts[0] in self.finger:
                return stmts[0]
            return self.visit(stmts[0], stmts[1:], break_point, continue_point)
        if continue_point:
            return continue_point
        return unbound_constant_expr()

    def visit_Return(
        self, node: ast.Return, _stmts, _break_point, _continue_point
    ):  # pylint: disable=invalid-name
        self.finger[node] = tuple()
        return node

    def visit_While(
        self, node: ast.While, stmts, break_point, continue_point
    ):  # pylint: disable=invalid-name
        self.finger[node] = None
        left = self.visit_stmts(node.body, stmts[0], node)
        right = self.visit_stmts(stmts, break_point, continue_point)
        self.finger[node] = (left, right)
        return node

    def visit_If(
        self, node: ast.If, stmts, break_point, continue_point
    ):  # pylint: disable=invalid-name
        self.finger[node] = None
        left = self.visit_stmts(node.body + stmts, break_point, continue_point)
        right = self.visit_stmts(node.orelse + stmts, break_point, continue_point)
        self.finger[node] = (left, right)
        return node

    def visit_Module(
        self, node: ast.Module, _stmts, break_point, continue_point
    ):  # pylint: disable=invalid-name
        return self.visit_stmts(node.body, break_point, continue_point)

    def visit_Assign(
        self, node: ast.Assign, stmts, break_point, continue_point
    ):  # pylint: disable=invalid-name
        self.finger[node] = []
        value = self.visit_stmts(stmts, break_point, continue_point)
        if value:
            self.finger[node] = (value,)
        return node

    def visit_Break(
        self, _node: ast.Break, _stmts, break_point, _continue_point
    ):  # pylint: disable=invalid-name,no-self-use
        return break_point

    def visit_Continue(
        self, _node: ast.Continue, _stmts, _break_point, continue_point
    ):  # pylint: disable=invalid-name,no-self-use
        return continue_point

    def visit_Pass(
        self, node: ast.Pass, stmts, break_point, continue_point
    ):  # pylint: disable=invalid-name
        self.finger[node] = None
        value = self.visit_stmts(stmts, break_point, continue_point)
        if value:
            self.finger[node] = (value,)
        return node

    def run(self, node: ast.AST) -> Tuple[ast.AST, Dict[ast.AST, Finger]]:
        self.finger = {}
        entry = self.visit(node, None, None, None)
        return entry, self.finger


class BasicBlock:  # pylint: disable=too-few-public-methods

    stmts: List[ast.AST]
    jumps: List["BasicBlock"]

    def __init__(self, stmt: ast.AST):
        self.stmts = []
        self.jumps = []
        if isinstance(stmt, (ast.Pass, ast.Break, ast.Continue)):
            return
        if isinstance(stmt, (ast.While, ast.If)):
            stmt = ast.If(test=stmt.test, body=None, orelse=None)
        self.stmts.append(stmt)


class CFG:

    bbs: List[BasicBlock]
    entry: BasicBlock
    bb2idx: Dict[BasicBlock, int]

    def __init__(self, bbs: List[BasicBlock], entry: BasicBlock, bb2idx: Dict[BasicBlock, int]):
        self.bbs = bbs
        self.entry = entry
        self.bb2idx = bb2idx

    def __str__(self):
        result = ""
        for bb in self.bbs:
            result += "BB {}:\n".format(self.bb2idx[bb])
            for stmt in bb.stmts:
                result += "  {}\n".format(stmt.__class__.__name__)
            for succ in bb.jumps:
                result += "  Jump: BB {}\n".format(self.bb2idx[succ])
        result += "Entry: BB {}\n".format(self.bb2idx[self.entry])
        return result

    @staticmethod
    def build(entry: ast.AST, finger: Dict[ast.AST, Finger]) -> "CFG":
        stmt2bb = {stmt: BasicBlock(stmt=stmt) for stmt in finger.keys()}
        for stmt, jumps in finger.items():
            stmt2bb[stmt].jumps = list(map(stmt2bb.get, jumps))
        bbs = list(stmt2bb.values())
        entry = stmt2bb[entry]
        bb2idx = {bb: idx for idx, bb in enumerate(bbs)}
        cfg = CFG(bbs=bbs, entry=entry, bb2idx=bb2idx)
        cfg.remove_empty()
        cfg.remove_dead()
        cfg.contract()
        cfg.remove_dead()
        cfg.sanity_check()
        return cfg

    def remove_empty(self) -> None:
        for bb in self.bbs:
            jumps = []
            for succ in bb.jumps:
                while not succ.stmts:
                    (succ,) = succ.jumps
                jumps.append(succ)
            bb.jumps = jumps

    def contract(self) -> None:
        in_degree = defaultdict(int)
        for bb in self.bbs:
            for succ in bb.jumps:
                in_degree[succ] += 1
        del_bbs = set()
        for bb in self.bbs:
            if bb in del_bbs:
                continue
            while len(bb.jumps) == 1:
                (succ,) = bb.jumps
                if succ is self.entry or in_degree.get(succ, 0) != 1:
                    break
                bb.stmts.extend(succ.stmts)
                bb.jumps = succ.jumps
                del_bbs.add(succ)
        self.bbs = [bb for bb in self.bbs if bb not in del_bbs]
        assert self.entry in self.bbs

    def remove_dead(self) -> None:
        visited = set([self.entry])
        queue = [self.entry]
        while queue:
            bb = queue.pop()
            for succ in bb.jumps:
                if succ in visited:
                    continue
                queue.append(succ)
                visited.add(succ)
        self.bbs = [bb for bb in self.bbs if bb in visited]
        assert self.entry in self.bbs

    def sanity_check(self) -> None:
        in_degree = defaultdict(int)
        for bb in self.bbs:
            for succ in bb.jumps:
                in_degree[succ] += 1
        for bb in self.bbs:
            try:
                # non-empty
                assert len(bb.stmts) >= 1
                # non-mergable
                if len(bb.jumps) == 1:
                    (succ,) = bb.jumps
                    assert succ is self.entry or in_degree.get(succ, 0) != 1
                # let-list
                for stmt in bb.stmts[:-1]:
                    assert isinstance(stmt, ast.Assign)
                # jump kind
                stmt = bb.stmts[-1]
                if not bb.jumps:
                    assert isinstance(stmt, ast.Return)
                elif len(bb.jumps) == 1:
                    assert isinstance(stmt, ast.Assign)
                elif len(bb.jumps) == 2:
                    assert isinstance(stmt, ast.If)
                else:
                    raise AssertionError
            except AssertionError:
                raise AssertionError("{}\nError on BB {}".format(self, self.bb2idx[bb]))


def ast2cfg(node: ast.AST) -> "CFG":
    entry, finger = FingerFinder().run(node)
    return CFG.build(entry, finger)
