/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/analysis/dependency_graph.h
 * \brief Utilities to manipulate and analyze dependency graph
 */
#pragma once

namespace mnm {
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
}  // namespace mnm
