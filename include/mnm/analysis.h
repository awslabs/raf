/*!
 * Copyright (c) 2021 by Contributors
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
 * \brief Create the dependency graph of given expr. We can prune the atomic nodes in the dependency
 *  graph. After pruning, the dependency graph only contains the nodes with expr type: Call, Tuple,
 *  and TupleGetItem.
 * \param arena Arena allocator used to allocate memory used by DependencyGraph.
 * \param e The expression we want to create dataflow graph for.
 * \param prune_atomic_node Whether to prune the atomic nodes.
 * \return The dataflow graph.
 */
DependencyGraph CreateDependencyGraph(Arena* arena, const Expr& e, bool prune_atomic_nodes = false);

}  // namespace analysis
}  // namespace mnm
