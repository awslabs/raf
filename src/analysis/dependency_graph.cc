/*!
 * Copyright (c) 2021 by Contributors
 * \file dataflow_graph.cc
 * \brief Create the dataflow graph from expr. The dataflow graph takes the operator/function call,
 * Tuple, and TupleGetItem as nodes and the dependency among them as edges. It is a directed acyclic
 * graph (DAG) and can be used to analyze the expr.
 */
#include "mnm/analysis.h"
#include "mnm/registry.h"

namespace mnm {
namespace analysis {

using Arena = tvm::support::Arena;
using Node = DependencyGraph::Node;
using LinkedList = tvm::relay::LinkedList<Node*>;
using NodeExprMap = std::unordered_map<const Node*, Expr>;

/*! Remove the edges between parent and child. */
void RemoveGraphEdge(Node* parent, Node* child) {
  auto remove_from_linklist = [](LinkedList* list, Node* value) {
    if (list->head->value == value) {
      list->head = list->head->next;
      if (list->head == nullptr) {
        list->tail = nullptr;
      }
    }
    for (auto p = list->head; p && p->next; p = p->next) {
      if (p->next->value == value) {
        p->next = p->next->next;
        if (p->next == nullptr) {
          list->tail = p;
        }
      }
    }
  };
  remove_from_linklist(&parent->children, child);
  remove_from_linklist(&child->parents, parent);
};

/*! A predicate function indicates whether a Node should be pruned. */
using FNodePredicate = std::function<bool(const Node*)>;

/*!
 * \brief Prune the nodes in a dependency graph, guided by a predicate.
 * \param dg The dependency graph on which we want to prune the nodes.
 * \param predicate If predicate(node) == True, we should prune this node and ALL its successors.
 */
void DependencyGraphPruneNodes(DependencyGraph* dg, const FNodePredicate& predicate) {
  std::set<Node*> nodes2remove;
  for (auto node : dg->post_dfs_order) {
    if (predicate(node) && nodes2remove.count(node) == 0) {
      // collect node and all its successors
      std::vector<Node*> stack({node});
      nodes2remove.insert(node);
      while (!stack.empty()) {
        Node* nd = stack.back();
        stack.pop_back();
        for (auto child = nd->children.head; child; child = child->next) {
          if (nodes2remove.count(child->value) == 0) {
            stack.push_back(child->value);
            nodes2remove.insert(child->value);
          }
        }
      }
    }
  }

  std::vector<std::pair<Node*, Node*>> edges2remove;
  for (auto node : nodes2remove) {
    for (auto parent = node->parents.head; parent; parent = parent->next) {
      edges2remove.emplace_back(parent->value, node);
    }
  }
  std::vector<Node*> new_order;
  for (auto node : dg->post_dfs_order) {
    if (!nodes2remove.count(node)) {
      new_order.push_back(node);
    }
  }
  dg->post_dfs_order = new_order;
  for (auto edge : edges2remove) {
    RemoveGraphEdge(edge.first, edge.second);
  }
}

/*!
 * \brief Prune the atomic nodes in the dependency graph.
 *
 * Notes
 * We assume that each node in the input dependency graph belongs to one of the following types:
 *   Call
 *   Tuple
 *   TupleGetItem
 *   Var
 *   GlobalVar
 *   Constant
 *   Op
 *   Function (with attr primitive = 1)
 * We consider the following nodes of these as **atomic nodes**:
 *   Var
 *   GlobalVar
 *   Constant
 *   Op
 *   Function (with attr primitive = 1)
 * We prune these nodes and their dependencies in the dependency graph because they do not involve
 * the execution flow of the model. After pruning, we can get a dependency graph with the following
 * nodes:
 *   Call
 *   Tuple
 *   TupleGetItem
 *
 * TODO(yaoyaoding): Add suuport of the following exprs.
 * We do not support the following exprs temporarily.
 *   If
 *   Let
 *   Function (with attr primitive = 0)
 *
 * We do not plan to support the following exprs.
 *   RefCreate
 *   RefRead
 *   RefWrite
 *   TempExprNode
 *   PrimFunc
 *   Constructor
 *
 * \param dg The dependency graph that is going to be pruned. The pruning takes in place on the
 *  input dependency graph.
 * \param node_expr The map from node in the dependency graph to its corresponding expr.
 *
 */
void DependencyGraphPruneAtomicNodes(DependencyGraph* dg, NodeExprMap node_expr) {
  auto predicate = [&](const Node* node) {
    Expr e = node_expr[node];
    return !e.defined() ||  // new scope node
           e->IsInstance<VarNode>() || e->IsInstance<GlobalVarNode>() ||
           e->IsInstance<RelayConstantNode>() || e->IsInstance<OpNode>() ||
           (e->IsInstance<FunctionNode>() &&
            e.as<FunctionNode>()->HasNonzeroAttr(attr::kPrimitive));
  };
  DependencyGraphPruneNodes(dg, predicate);
}

/*!
 * \brief Get the nodes and edges of given expr's dependency graph
 * \param e The expr for which we want to get the dataflow graph
 * \param keep_atomic_nodes Whether to keep the atomic nodes in the result graph.
 * \return A map {"nodes": nodes, "edges": edges}. Here nodes is an array of expr, representing the
 * nodes in the dataflow graph. And edges are an array of expr pairs, representing the edges in the
 * dataflow graph.
 */
Map<String, ObjectRef> GetDependencyGraphNodesEdges(Expr e, bool prune_atomic_nodes) {
  Arena arena;
  DependencyGraph graph;

  graph = CreateDependencyGraph(&arena, e, prune_atomic_nodes);
  NodeExprMap node_expr;
  for (auto& it : graph.expr_node) {
    node_expr[it.second] = it.first;
  }

  Array<Expr> nodes;
  Array<Array<Expr>> edges;
  for (auto node : graph.post_dfs_order) {
    nodes.push_back(node_expr[node]);
  }
  for (auto node : graph.post_dfs_order) {
    Expr parent = node_expr[node];
    for (auto child_iter = node->children.head; child_iter; child_iter = child_iter->next) {
      Expr child = node_expr[child_iter->value];
      edges.push_back({parent, child});
    }
  }
  return {{"nodes", nodes}, {"edges", edges}};
}

DependencyGraph CreateDependencyGraph(Arena* arena, const Expr& e, bool prune_atomic_nodes) {
  DependencyGraph dg = DependencyGraph::Create(arena, e);

  if (prune_atomic_nodes) {
    NodeExprMap node_expr;
    for (auto& it : dg.expr_node) {
      node_expr[it.second] = it.first;
    }

    DependencyGraphPruneAtomicNodes(&dg, node_expr);
  }

  return std::move(dg);
}

MNM_REGISTER_GLOBAL("mnm.analysis.GetDependencyGraphNodesEdges")
    .set_body_typed(GetDependencyGraphNodesEdges);

}  // namespace analysis
}  // namespace mnm
