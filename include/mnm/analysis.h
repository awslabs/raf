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
 * \file include/analysis.h
 * \brief Analysis used by passes.
 */
#pragma once
#include <relay/analysis/dependency_graph.h>
#include "support/arena.h"

#include "mnm/ir.h"

namespace mnm {
namespace analysis {

using namespace mnm::ir;
using tvm::relay::DependencyGraph;
using tvm::support::Arena;

/*!
 * \brief Create the dependency graph of given expr.
 *
 * We can prune the atomic nodes in the dependency graph. After pruning, the dependency graph only
 * contains the nodes with expr type: Call, Tuple, and TupleGetItem.
 *
 * We can also prune the redundant edges in the dependency graph. An edge (u, v) is redundant if
 * and only if there exists a path from u to v that does not go through edge (u, v) directly. We
 * call such edge "redundant" because the dependency relation has be indicated by the path.
 *
 * \param arena Arena allocator used to allocate memory used by DependencyGraph.
 * \param e The expression we want to create dataflow graph for.
 * \param prune_atomic_node Whether to prune the atomic nodes.
 * \param prune_redundant_edges Whether to prune the redundant edges.
 * \return The dataflow graph.
 */
DependencyGraph CreateDependencyGraph(Arena* arena, const Expr& e, bool prune_atomic_nodes = false,
                                      bool prune_redundant_edges = false);

}  // namespace analysis
}  // namespace mnm
