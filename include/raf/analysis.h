/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file include/analysis.h
 * \brief Analysis used by passes.
 */
#pragma once
#include <relay/analysis/dependency_graph.h>
#include "support/arena.h"

#include "raf/ir_ext.h"

namespace raf {
namespace analysis {

using namespace raf::ir;
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
}  // namespace raf
