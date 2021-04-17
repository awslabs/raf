/*!
 * Copyright (c) 2019 by Contributors
 * \file pass.h
 * \brief A compatibility layer for TVM/Relay passes
 */
#pragma once

#include "tvm/ir/transform.h"
#include "tvm/relay/analysis.h"
#include "tvm/relay/dataflow_matcher.h"
#include "tvm/relay/transform.h"
#include "mnm/ir.h"
#include "mnm/ir_ext.h"
#include "mnm/pass_manager.h"

namespace mnm {
namespace pass {
using tvm::AsText;
using tvm::relay::ExpandANormalForm;
using tvm::relay::FreeVars;
using tvm::transform::CreateModulePass;
using tvm::transform::Pass;
using tvm::transform::PassContext;
using tvm::transform::PassInfo;
/*!
 * \brief A pass that does automatic differentiation.
 * \param requires_grads If input(s) of function requires gradient. It is in the same order as
 * func->param. If empty, input(s) with float datatype requires gradient.
 * \return The created passed.
 */
Pass AutoDiff(ir::Array<tvm::Bool> requires_grads = {});
/*!
 * \brief A pass that performs data parallelism. It mainly modifies the backward
 * closure by adding communication ops after the ops that generate local
 * gradient and stream_sync ops before the end of backward closure to ensure
 * communication is done.
 * \return The created pass.
 */
Pass AutoDataParallel();
/*!
 * \brief The constant folding pass.
 * \return The created pass.
 */
Pass FoldConstant();
ir::Expr BindParam(ir::Function func, ir::Array<ir::Expr> args);
/*!
 * \brief A pass that lifts the lambda to the global scope.
 * \return The created pass.
 */
Pass LambdaLift();
/*!
 * \brief A pass that is used for gradient operator input selection.
 * \return The created pass.
 */
Pass GradInputSelect();
/*!
 * \brief A pass that manifests memory allocation.
 * \return The created pass.
 */
Pass ManifestAlloc();
/*!
 * \brief A pass that canonicalize operators.
 * \return The created pass.
 */
Pass CanonicalizeOps();
/*!
 * \brief Create a type inference pass.
 * \return The created pass.
 */
Pass InferType();
ir::Expr InferType(ir::Expr expr);
/*!
 * \brief A pass that removes unnecessary memory allocation and perform inplace updates.
 * \return The created pass.
 */
Pass InplaceUpdate();

/*!
 * \brief Create a pass to wrap an expr with compiler_begin and compiler_end to indicate that this
 * expr should be handled by the external compiler.
 * \param target he target backends for annotation.
 * \return The pass.
 */
Pass AnnotateTarget(ir::Array<ir::String> target);

/*!
 * \brief After operators have been annotated with the targets that support
 * them, this pass creates regions of the operators for each target. It
 * is guaranteed that the regions will have a topological rodering so that
 * no data dependency issue exist.
 * \return The created pass.
 */
Pass MergeCompilerRegions();

/*!
 * \brief Partition an input function into multiple functions according based
 * on the inserted annotation nodes (i.e. compiler_begin and compiler_end).
 * These nodes are used as boundaries to partition the Relay function into
 * multiple regions that can be offloaded to different accelerators/backends.
 * \return The created pass.
 */
Pass PartitionGraph();

/*!
 * \brief A pass that casts input(s) of some operators in the expression.
 * \return The created pass.
 */
Pass AutoCast();

/*!
 * \brief A pass that inlines the Let stmt that assigns a var to another and TupleGetItem that can
 * be simplified.
 * \return The created pass.
 */
Pass InlineLet();

/*! \brief A pass that removes expressions which does not effect the program result.
 *
 * It will remove let bindings which are not referenced.
 *
 * For example, this pass should turn `let a = 1 in 2` into `2`,
 * as the value of the expression does not depend on a.
 *
 * As another example, `let a = 1 in a` will be optimized into 1.
 *
 * \return The created pass.
 */
Pass DeadCodeElimination();

/*!
 * \brief A pass that simplifies commonly seen patterns that can be removed at compile time.
 * \return The created pass.
 */
Pass SimplifyExpr();

/*! \brief Convert Relay IR to Meta IR.
 * \param disabled_pass A list of pass names to be disabled.
 * \return The created pass.
 */
Pass FromRelay(ir::Array<tvm::String> disabled_pass = {});

/*!
 * \brief A pass that inlines backward function.
 * \return The created pass.
 */
Pass InlineBackward();

/*!
 * \brief Substitute variables in expr
 * \param expr The expression
 * \param args_map The substitution rule
 * \return Transformed expression
 */
ir::Expr Substitute(ir::Expr expr, const tvm::Map<ir::Var, ir::Expr>& args_map);

/*!
 * \brief A pass that replaces init and constant ops with the assigned device.
 * \param device The target device.
 * \return The created pass.
 */
Pass AssignDevice(std::string device);

/*!
 * \brief A pass that lifts if true and false branches to global functions.
 * \return The created pass.
 */
Pass LiftBranchBody();

/*!
 * \brief This pass is applied after Lambda lifting. Lambda lifting pass lifts the closures to
 * global scope, but the lifted global function still has the closure within. This makes AD harder.
 * This pass flattens the global functions that are marked Closure, and then changes the call sites
 * accordingly. This helps AD pass where it is difficult to handle closures.
 * \return The created pass.
 */
Pass FlattenClosure();
/*!
 * \brief Performs operator fusion.
 * \param mod IRModule to be fused.
 * \param fuse_opt_level The optimization level used to enable this pass.
 * \return The created pass.
 *
 */
Pass FuseOps(int fuse_opt_level);
/*!
 * \brief A pass that eliminates dead code.
 * \return The created pass.
 */
Pass DeadCodeElimination();
/*!
 * \brief A pass that convert A-normal form to dataflow graph.
 * \return The created pass.
 */
Pass ToGraphNormalForm();
/*!
 * \brief A pass that turns a dataflow graph into Administrative Normal Form, or A-Normal Form
 * (ANF).
 *
 * It will turn an expression that is in a graph form (with sharing implicit),
 * to an expression with explicit sharing (A-Normal Form).
 *
 * The scope of the root expression is the global scope.
 *
 * The scope of any non root expression is the least common ancestor of all it's scope.
 *
 * Values are ordered by post-DFS order in each scope.
 *
 * \return The created pass.
 */
Pass ToANormalForm();

TVM_DLL Pass CreateMNMFunctionPass(
    const runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)>& pass_func,
    int opt_level, String name, tvm::Array<String> required);

/*!
 * \brief A pass that turns an expression to Basic Block Normal Form.
 *
 * We define a block as a group of expressions implied by the scope structure.
 *
 * Each graph node can only belong to a single block.
 *
 * For any value that is being used in multiple blocks, it has to be referred
 * by a Var which is defined in a block, whose scope is the least common ancestor
 * of blocks this value is used.
 *
 * \return The created pass.
 */
Pass ToBasicBlockNormalForm();

}  // namespace pass
}  // namespace mnm
