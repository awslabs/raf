/*!
 * Copyright (c) 2019 by Contributors
 * \file pass.h
 * \brief A compatibility layer for TVM/Relay passes
 */
#pragma once

#include "tvm/relay/analysis.h"
#include "tvm/relay/transform.h"
#include "mnm/ir.h"
#include "mnm/ir_ext.h"

namespace mnm {
namespace pass {
using tvm::AsText;
using tvm::relay::FreeVars;
ir::Function AutoDiff(ir::Function func);
ir::Function AutoDataParallel(ir::Function func);
ir::Expr FoldConstant(ir::Expr expr, ir::Module mod);
ir::Expr BindParam(ir::Function func, ir::Array<ir::Expr> args);
ir::Module LambdaLift(ir::Module mod);
/*!
 * \brief Manifest memory allocation.
 * \param mod The IR module.
 * \return Transformed IR module.
 */
ir::Module ManifestAlloc(ir::Module mod);
ir::Expr CanonicalizeOps(ir::Expr expr);
ir::Module InferType(ir::Module mod);
ir::Expr InferType(ir::Expr expr);
/*!
 * \brief Fuse the operators in the expression.
 * \param expr Expression to be fused.
 * \param fuse_opt_level Optimization level. If it is 0, then no operators will be fused.
 * \return Transformed expression.
 */
ir::Expr FuseOps(ir::Expr expr, int fuse_opt_level);

/*!
 * \brief remove unnecessary memory allocation and perform inplace updates.
 * \param mod The IR module.
 * \return Transformed IR module.
 */
ir::Module InplaceUpdate(ir::Module mod);

/*!
 * \brief Wraps an expr with compiler_begin and compiler_end to indicate that
 * this expr should be handled by the external compiler.
 * \param expr Expression to be annotated.
 * \param target he target backends for annotation.
 * \return Transformed Expression.
 */
ir::Expr AnnotateTarget(ir::Expr expr, ir::Array<ir::String> target);

/*!
 * \brief After operators have been annotated with the targets that support
 * them, this pass creates regions of the operators for each target. It
 * is guaranteed that the regions will have a topological rodering so that
 * no data dependency issue exist.
 *
 * This pass only introduces annotations to indicate the regions.
 * partition_graph must subsequently be called to lift these regions out
 * as external functions.
 * \param expr Expression to be merged.
 * \return Transformed Expression.
 */
ir::Expr MergeCompilerRegions(ir::Expr expr);

/*!
 * \brief Partition an input function into multiple functions according based
 * on the inserted annotation nodes (i.e. compiler_begin and compiler_end).
 * These nodes are used as boundaries to partition the Relay function into
 * multiple regions that can be offloaded to different accelerators/backends.
 *
 * Each of these paritioned functions, a.k.a regions, will be viewed as
 * external functions, and they will use the provided compiler for codegen.
 * \param expr Expression to be partition.
 * \return Parartioned Expression.
 */
ir::Expr PartitionGraph(ir::Expr expr);

/*!
 * \brief Cast input(s) of some operators in the expression.
 * \param expr Expression to be casted.
 * \return Transformed Expression.
 */
ir::Expr AutoCast(ir::Expr func);

/*!
 * \brief Inline the Let stmt that assigns a var to another and TupleGetItem that can be simplified.
 * \param expr Expression to be inlined.
 * \return Transformed Expression.
 */
ir::Expr InlineLet(ir::Expr expr);

/*! \brief Remove expressions which does not effect the program result.
 *
 * It will remove let bindings which are not referenced.
 *
 * For example, this pass should turn `let a = 1 in 2` into `2`,
 * as the value of the expression does not depend on a.
 *
 * As another example, `let a = 1 in a` will be optimized into 1.
 *
 * \param expr Expression to be transformed.
 *
 * \return Transformed Expression.
 */
ir::Expr DeadCodeElimination(const ir::Expr& expr);

/*! \brief Convert Relay IR to Meta IR.
 * \param obj tvm::IRModule or ir::Expr
 * \return ir::Module or ir::Expr
 */
tvm::ObjectRef FromRelay(tvm::ObjectRef obj);

}  // namespace pass
}  // namespace mnm
