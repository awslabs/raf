/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/analysis/dependency_graph.h
 * \brief Utilities to manipulate and analyze dependency graph
 */
#pragma once

namespace raf {
namespace analysis {
namespace dependency_graph {

using Arena = tvm::support::Arena;
using Node = DependencyGraph::Node;
using NodeList = tvm::relay::LinkedList<Node*>;
using NodeExprMap = std::unordered_map<const Node*, Expr>;

/*!
 * \brief Remove the edge between parent and child. If there are duplicated edges between them,
 * the duplicated edges will also be removed.
 * \param parent The parent node of the edge.
 * \param child The child node of the edge.
 */
void RemoveGraphEdge(Node* parent, Node* child);

/*!
 * Add an edge between parent and child
 * \param arena The arena to allocate memory.
 * \param parent The parent node of the edge.
 * \param child The child node of the edge.
 */
void AddGraphEdge(Arena* arena, Node* parent, Node* child);

/*!
 * Get a topological order of the graph derived from given nodes.
 * \param nodes The nodes to get the topological order from.
 * \return A topological order of nodes.
 */
std::vector<Node*> GetTopologicalOrder(const std::vector<Node*>& nodes);

/*!
 * \brief Get the number of nodes in the linked list.
 * \return The number of nodes.
 */
size_t GetListSize(const tvm::relay::LinkedList<Node*>& node_list);

/*!
 * Create a new node.
 * \param arena The arena to allocate memory.
 * \return The created node.
 */
Node* CreateNewNode(Arena* arena);

}  // namespace dependency_graph
}  // namespace analysis
}  // namespace raf
