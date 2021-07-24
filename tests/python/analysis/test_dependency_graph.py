# pylint: disable=invalid-name, no-self-use, protected-access
import mnm
from mnm.testing import randn
from mnm._ffi.pass_ import ToGraphNormalForm
from mnm._ffi.analysis import GetDependencyGraphNodesEdges


def test_pruning():
    class Model(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x):
            p0 = mnm.atan(x)   # op 1

            p1 = mnm.atan(x)   # op 2
            p1 = mnm.atan(p1)  # op 3

            p2 = mnm.atan(x)   # op 4
            p2 = mnm.atan(p2)  # op 5
            p2 = mnm.atan(p2)  # op 6
            return mnm.concatenate([p0, p1, p2])  # op 7
    model = Model()
    input_shape = [2, 2]
    x, _ = randn(input_shape)
    mod = model._internal(x).mod

    mod = ToGraphNormalForm()(mod)

    expr = mod['main'].body
    original_graph = GetDependencyGraphNodesEdges(expr, False)
    pruned_graph = GetDependencyGraphNodesEdges(expr, True)
    original_num_edges = len(original_graph["edges"])
    pruned_num_edges = len(pruned_graph["edges"])
    # Please use the mnm.utils.dependency_graph_visualizer.draw_dependency_graph to draw the
    # dependency graph.
    assert original_num_edges == 18
    assert pruned_num_edges == 7
