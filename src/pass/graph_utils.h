/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file graph_utils.h
 * \brief Utilities for generating dominator tree graph.
 */
#pragma once

#include <support/arena.h>
#include "raf/ir.h"
#include "raf/op.h"

namespace raf {
namespace pass {

using namespace raf::ir;
using namespace raf::op;
using namespace tvm::support;

/*!
 * \brief Indexed data flow graph in forward direction.
 *  This is a temporary data structure used for operator fusion analysis.
 *
 *  This data structure only captures the dataflow fragment and
 *  could ignore blocks like let by simply ordering each dataflow block
 *  and mark the output node as extern_ref;
 */
class IndexedForwardGraph {
 public:
  struct Node;
  /*!
   * The forward edge in the dataflow graph.
   */
  struct Edge {
    /*! \brief The corresponding node */
    Node* node{nullptr};
    /*! \brief The respective pattern of this op */
    OpPatternKind pattern{kOpaque};
  };
  /*! \brief A node in the graph. */
  struct Node {
    /*! \brief weak reference to the corresponding edge. */
    const tvm::Object* ref{nullptr};
    /*! \brief The index of the node in topological order. */
    size_t index{0};
    /*! \brief Whether this node is referenced by external source */
    bool extern_ref{false};
    /*! \brief The general pattern in the node */
    OpPatternKind pattern{kOpaque};
    /*! \brief The outputs of the node. */
    LinkedList<Edge> outputs;
    /*! \brief Whether any outputs have inplace update. */
    bool inplace_update{false};
    /*! \brief Dump the node into string. */
    std::string DebugDump() const;
  };
  /*! \brief The node map that maps node to graph */
  std::unordered_map<const tvm::Object*, Node*> node_map;
  /*! \brief All the nodes in post DFS order */
  std::vector<Node*> post_dfs_order;
  /*! \brief Dump the graph into string. */
  std::string DebugDump() const;
  /*!
   * \brief create a indexed forward graph.
   * \param arena The arena used for data allocation.
   * \param body The body of the expression to create a graph.
   */
  static IndexedForwardGraph Create(Arena* arena, const Expr& body);

 private:
  class Creator;
};

/*!
 * \brief Dominator tree that represent domination or
 *  post domination relation of the node.
 */
class DominatorTree {
 public:
  /*!
   * \brief A node in the dominator tree.
   */
  struct Node {
    /*! \brief The node in the tree */
    IndexedForwardGraph::Node* gnode{nullptr};
    /*! \brief parent of the tree */
    Node* parent{nullptr};
    /*! \brief current depth*/
    int depth{0};
    /*! \brief aggregated pattern to parent */
    OpPatternKind pattern{kOpaque};
    /*! \brief Dump the node into string. */
    std::string DebugDump() const;
  };
  // index -> node.
  std::vector<Node*> nodes;
  /*!
   * \brief compute a post dominator relation for a given dataflow graph.
   * \param arena The arena used for node allocation.
   * \param graph The graph to be analyzed.
   * \return The dominator tree of the graph.
   * \note This algorithm makes use of the fact that graph is DAG,
   *       and runs a single pass algorithm via LCA (Least Common Ancestor)
   */
  static DominatorTree PostDom(Arena* arena, const IndexedForwardGraph& graph);
  /*! \brief Dump all the nodes in the tree into string. */
  std::string DebugDump() const {
    std::ostringstream os;
    for (int i = 0; i < nodes.size(); ++i) {
      os << "treenode[" << i << "]: " << nodes[i]->DebugDump() << "\n";
    }
    return os.str();
  }

 private:
  // Combine pattern together.
  static OpPatternKind CombinePattern(OpPatternKind lhs, OpPatternKind rhs) {
    if (lhs > rhs) return lhs;
    return rhs;
  }
  /*!
   * \brief Find the least common ancestor of the two nodes.
   * \param lhs The left node.
   * \param rhs The right node.
   * \param edge_pattern
   *        The combined edge pattern across all the parents.
   * \return The least common ancestor of the two.
   */
  static Node* LeastCommonAncestor(Node* lhs, Node* rhs, OpPatternKind* edge_pattern);
  /*!
   * \brief Find the least common ancestor of a list of nodes.
   * \param nodes the nodes.
   * \param edge_pattern
   *        The combined edge pattern across all the parents.
   * \return The least common ancestor of all nodes.
   */
  Node* LeastCommonAncestor(const LinkedList<IndexedForwardGraph::Edge>& input_nodes,
                            OpPatternKind* edge_pattern);
  /*!
   * \brief Convert the Node from an IndexedForwardGraph Node into DomaintorTree Node.
   * \param arena The Arena.
   * \param gnode An IndexedForwardGraph Node.
   * \return The DominatorTree Node.
   */
  Node* GetNode(Arena* arena, IndexedForwardGraph::Node* gnode);
};

}  // namespace pass
}  // namespace raf
