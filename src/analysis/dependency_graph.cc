/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file dataflow_graph.cc
 * \brief Create the dataflow graph from expr. The dataflow graph takes the operator/function call,
 * Tuple, and TupleGetItem as nodes and the dependency among them as edges. It is a directed acyclic
 * graph (DAG) and can be used to analyze the expr.
 */
#include "support/arena.h"
#include "raf/analysis.h"
#include "raf/registry.h"

#include "./dependency_graph.h"

namespace raf {
namespace analysis {
namespace dependency_graph {

using tvm::support::LinkNode;

/*! Remove the edges between parent and child. */
void RemoveGraphEdge(Node* parent, Node* child) {
  auto remove_from_nodelist = [](NodeList* list, Node* value) {
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
  remove_from_nodelist(&parent->children, child);
  remove_from_nodelist(&child->parents, parent);
};

void AddGraphEdge(Arena* arena, Node* parent, Node* child) {
  auto* parent_link = arena->make<LinkNode<Node*>>();
  parent_link->value = parent;
  child->parents.Push(parent_link);

  auto* child_link = arena->make<LinkNode<Node*>>();
  child_link->value = child;
  parent->children.Push(child_link);
}

/*! \brief Get the size (length) of a linked list. */
size_t GetListSize(const tvm::relay::LinkedList<Node*>& list) {
  if (list.head == nullptr) return 0;
  size_t size = 0;
  for (auto p = list.head; p; p = p->next) {
    size++;
  }
  return size;
}

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
  std::vector<Node*> new_post_dfs_order;
  for (auto node : dg->post_dfs_order) {
    if (!nodes2remove.count(node)) {
      new_post_dfs_order.push_back(node);
    }
  }
  dg->post_dfs_order = new_post_dfs_order;

  std::unordered_set<Node*> existing_nodes(new_post_dfs_order.begin(), new_post_dfs_order.end());
  std::vector<Expr> expr2remove;
  for (auto& pr : dg->expr_node) {
    if (existing_nodes.count(pr.second) == 0) {
      expr2remove.push_back(pr.first);
    }
  }
  for (const Expr& e : expr2remove) {
    dg->expr_node.erase(e);
  }

  std::vector<std::pair<Node*, Node*>> edges2remove;
  for (auto node : nodes2remove) {
    for (auto parent = node->parents.head; parent; parent = parent->next) {
      edges2remove.emplace_back(parent->value, node);
    }
  }
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
void DependencyGraphPruneAtomicNodes(DependencyGraph* dg) {
  NodeExprMap node_expr;
  for (auto& it : dg->expr_node) {
    node_expr[it.second] = it.first;
  }

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
 * \brief Prune the redundant edges in the dependency graph. One edge (u, v) is redundant if and
 * only if there exists a path from u to v that does not go through the edge (u, v) directly. We
 * call the edge "redundant" because the dependency relation has been indicated by the path.
 *
 * The time complexity is O(NE), where N is the number of nodes and E is the number of edges in
 * the dependency graph, which may be slow for large complete graph. But it is efficient enough for
 * almost all neural networks.
 *
 * \param dg The dependency graph.
 */
void DependencyGraphPruneRedundantEdges(DependencyGraph* dg) {
  std::unordered_map<const Node*, std::unordered_set<const Node*>> indirect_children_map;
  std::vector<std::pair<Node*, Node*>> edges2remove;

  std::unordered_map<const Node*, int> node_index;
  for (const Node* node : dg->post_dfs_order) {
    node_index[node] = int(node_index.size());
  }

  for (const Node* node : dg->post_dfs_order) {
    for (auto iit = node->parents.head; iit; iit = iit->next) {
      Node* parent = iit->value;
    }
  }

  for (Node* node : dg->post_dfs_order) {
    auto& indirect_children = indirect_children_map[node];
    for (auto iit = node->children.head; iit; iit = iit->next) {
      Node* child = iit->value;
      if (indirect_children.count(child)) {
        // There is a path from child to node that does not go through edge (child, node) directly.
        edges2remove.emplace_back(child, node);
      }
    }
    for (auto iit = node->children.head; iit; iit = iit->next) {
      Node* child = iit->value;
      indirect_children.insert(child);
    }
    for (auto iit = node->parents.head; iit; iit = iit->next) {
      Node* parent = iit->value;
      auto& parent_indirect_children = indirect_children_map[parent];
      for (const Node* indirect_child : indirect_children) {
        parent_indirect_children.insert(indirect_child);
      }
    }
  }

  for (auto& pr : edges2remove) {
    Node *parent, *child;
    std::tie(child, parent) = pr;
    RemoveGraphEdge(parent, child);
  }
}

std::vector<Node*> GetTopologicalOrder(const std::vector<Node*>& nodes) {
  std::unordered_set<Node*> nodes_set(nodes.begin(), nodes.end());
  std::unordered_map<Node*, int> out_degree;
  std::vector<Node*> post_dfs_order;
  for (auto node : nodes) {
    for (auto iit = node->parents.head; iit; iit = iit->next) {
      auto parent = iit->value;
      CHECK_NE(nodes_set.count(parent), 0);
      out_degree[parent]++;
    }
  }
  std::vector<Node*> stack;
  for (auto node : nodes) {
    if (out_degree[node] == 0) {
      stack.push_back(node);
    }
  }
  while (!stack.empty()) {
    Node* node = stack.back();
    stack.pop_back();
    post_dfs_order.push_back(node);
    for (auto iit = node->parents.head; iit; iit = iit->next) {
      auto parent = iit->value;
      if (--out_degree[parent] == 0) {
        stack.push_back(parent);
      }
    }
  }
  return post_dfs_order;
}

Node* CreateNewNode(Arena* arena) {
  Node* node = arena->make<DependencyGraph::Node>();
  node->new_scope = false;
  return node;
}

}  // namespace dependency_graph

DependencyGraph CreateDependencyGraph(Arena* arena, const Expr& e, bool prune_atomic_nodes,
                                      bool prune_redundant_edges) {
  DependencyGraph dg = DependencyGraph::Create(arena, e);

  if (prune_atomic_nodes) {
    dependency_graph::DependencyGraphPruneAtomicNodes(&dg);
  }

  if (prune_redundant_edges) {
    dependency_graph::DependencyGraphPruneRedundantEdges(&dg);
  }

  return std::move(dg);
}

/*!
 * \brief Get the nodes and edges of given relay expression's dependency graph
 * \param e The expr for which we want to get the dataflow graph
 * \param keep_atomic_nodes Whether to keep the atomic nodes in the result graph.
 * \return A map {"nodes": nodes, "edges": edges}. Here nodes is an array of expr, representing the
 * nodes in the dataflow graph. And edges are an array of expr pairs, representing the edges in the
 * dataflow graph.
 */
Map<String, ObjectRef> GetDependencyGraphNodesEdges(Expr e, bool prune_atomic_nodes,
                                                    bool prune_redundant_edges) {
  Arena arena;
  DependencyGraph graph;

  graph = CreateDependencyGraph(&arena, e, prune_atomic_nodes, prune_redundant_edges);
  dependency_graph::NodeExprMap node_expr;
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

RAF_REGISTER_GLOBAL("raf.analysis.GetDependencyGraphNodesEdges")
    .set_body_typed(GetDependencyGraphNodesEdges);

}  // namespace analysis
}  // namespace raf
