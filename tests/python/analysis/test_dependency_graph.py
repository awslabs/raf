# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=no-self-use, protected-access
import pytest
import raf
from raf.testing import randn
from raf._ffi.pass_ import ToGraphNormalForm
from raf._ffi.analysis import GetDependencyGraphNodesEdges


def test_prune_atomic_nodes():
    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            p_0 = raf.atan(x)  # atan 1

            p_1 = raf.atan(x)  # atan 2
            p_1 = raf.atan(p_1)  # atan 3

            p_2 = raf.atan(x)  # atan 4
            p_2 = raf.atan(p_2)  # atan 5
            p_2 = raf.atan(p_2)  # atan 6
            tup = [p_0, p_1, p_2]  # tup
            return raf.concatenate(tup)  # concat

    model = Model()
    input_shape = [2, 2]
    x, _ = randn(input_shape)
    mod = model._internal(x).mod

    mod = ToGraphNormalForm()(mod)

    expr = mod["main"].body
    original_graph = GetDependencyGraphNodesEdges(expr, False, False)
    original_num_edges = len(original_graph["edges"])
    assert original_num_edges == 18

    pruned_graph = GetDependencyGraphNodesEdges(expr, True, False)
    pruned_num_edges = len(pruned_graph["edges"])
    # Edges:
    # atan 2 -> atan 3
    # atan 4 -> atan 5
    # atan 5 -> atan 6
    # atan 1 -> tup
    # atan 3 -> tup
    # atan 6 -> tup
    # tup -> concat
    assert pruned_num_edges == 7


def test_prune_redundant_edges():
    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            y = raf.atan(x)  # atan 0
            z = raf.atan(y)  # atan 1
            tup = [y, z]  # tup
            return raf.concatenate(tup)  # concat

    model = Model()
    input_shape = [2, 2]
    x, _ = randn(input_shape)
    mod = model._internal(x).mod

    mod = ToGraphNormalForm()(mod)

    expr = mod["main"].body
    graph = GetDependencyGraphNodesEdges(expr, True, False)
    num_edges = len(graph["edges"])
    # Edges:
    # atan 0 -> atan 1
    # atan 0 -> tup
    # atan 1 -> tup
    # tup -> concat
    assert num_edges == 4

    pruned_graph = GetDependencyGraphNodesEdges(expr, True, True)
    pruned_num_edges = len(pruned_graph["edges"])
    # Edges:
    # atan 0 -> atan 1
    # atan 1 -> tup
    # tup -> concat
    # Pruned edge:
    # atan 0 -> tup
    assert pruned_num_edges == 3


if __name__ == "__main__":
    pytest.main([__file__])
