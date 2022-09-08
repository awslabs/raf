/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file pass.h
 * \brief A compatibility layer for TVM/Relay passes
 */
#pragma once

#include "tvm/ir/transform.h"
#include "tvm/relay/analysis.h"
#include "tvm/relay/dataflow_matcher.h"
#include "tvm/relay/transform.h"
#include "raf/ir_ext.h"
#include "raf/value.h"
#include "raf/ir_ext.h"
#include "raf/pass_manager.h"

namespace raf {
namespace pass {

using tvm::relay::ExpandANormalForm;
using tvm::relay::FreeTypeVars;
using tvm::relay::FreeVars;
using tvm::relay::Function;
using tvm::runtime::PackedFunc;
using tvm::runtime::TypedPackedFunc;
using tvm::transform::CreateModulePass;
using tvm::transform::Pass;
using tvm::transform::PassContext;
using tvm::transform::PassInfo;
using tvm::transform::PassNode;

/*!
 * \brief Create a function pass.
 * \param pass_func The packed function that contains the optimization.
 * \param opt_level The optimization level of the function pass.
 * \param name The name of the function pass.
 * \param required The list of the passes that the function pass is dependent on.
 * \return The created function pass.
 */
TVM_DLL Pass
CreateRAFFunctionPass(const TypedPackedFunc<Function(Function, IRModule, PassContext)>& pass_func,
                      int opt_level, String name, tvm::Array<String> required);

/*!
 * \brief A special trace pass that prints the header and IR to LOG(INFO).
 * \param header The header to be attached to the output.
 * \param show_meta_data Whether should we show meta data.
 * \return The pass.
 */
Pass PrintIR(const std::string& header = "", bool show_meta_data = false);

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

/*!
 * \brief A pass that lifts the lambda to the global scope.
 * \return The created pass.
 */
Pass LambdaLift();

/*!
 * \brief A pass that inlines global functions and erases the inlined functions
 * from the module.
 * \return The created pass.
 */
Pass FullInline();

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
 * \brief A pass that optimizes memory footprint.
 * \return The created pass.
 */
Pass MemoryPlan();

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

/*!
 * \brief Create a type erasing pass.
 * \return The created pass.
 */
Pass EraseType();

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
 * \brief A pass that rematerializes tensors to reduce memory footprint.
 * \return The created pass.
 */
Pass Rematerialization();

/*!
 * \brief A pass that schedules ANF for memory optimization.
 * \return The created pass.
 */
Pass MemorySchedule();

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

/*! \brief Convert Relay IR to RAF IR.
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
 * \brief Fuse the operators based on registered dialect fusion patterns.
 * \return The created pass.
 */
Pass FuseDialect();

/*!
 * \brief Performs operator fusion using TVM.
 * \return The created pass.
 */
Pass FuseTVM();

/*!
 * \brief Dispatch the base operators to dialect operators based on plevel.
 * \return The created pass.
 */
Pass DispatchDialect();

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

/*!
 * \brief Inline the primitive ops into the call position in the A-normal form.
 * \return The created pass.
 */
Pass InlinePrimitives();

/*!
 * \brief Inline closures
 * \return The created pass.
 */
Pass InlineClosure();

/*!
 * \brief This pass marks the may_share in the variables for ops that have attr TRAFInplaceUpdate
 * indicating the inputs and outputs to share the memory.
 * \return The created pass.
 */
Pass InplaceUpdate();

/*!
 * TODO(@hzfan): Update the doc for enforce_memory_share
 * \brief This pass validates and corrects the memory share annotated by the user.
 * \param enforce_inplace_update TBD.
 * \return The created pass.
 */
Pass ValidateInplaceUpdate(bool enforce_inplace_update);

/*!
 * \brief This pass implements wavefront stream schedule policy. It transforms BBNF into ANF and
 * injects stream-related operators (e.g., raf.op.set_stream, raf.op.add_event, and
 * raf.op.wait_event).
 * \return The created pass.
 */
Pass WavefrontStreamSchedule();

/*!
 * \brief This pass implements ASAP (as soon as possible) stream schedule policy. It transforms
 * BBNF into ANF and injects stream-related operators (e.g., raf.op.set_stream, raf.op.add_event,
 * and raf.op.wait_event).
 * \return The created pass.
 */
Pass ASAPStreamSchedule();

/*!
 * \brief This pass transforms BBNF into ANF and schedules operators to improve overlapping
 * between computation and communication.
 * \return The created pass.
 */
Pass DataParallelSchedule();

/*!
 * \brief This pass works in ANF and adds necessary synchronization ops (i.e., raf.op.set_stream,
 * raf.op.add_event, and raf.op.wait_event) between communication ops and computation ops to
 * ensure correctness. This pass must be run if AutoDataParallel is enabled.
 * \return The created pass.
 */
Pass EnforceSync();

/*!
 * \brief This pass works in ANF and adds neccessary memory copy ops before and after
 * multi-input collectives ops to pipeline memory copies.
 * \return The created pass.
 */
Pass AnnotateCollectiveOps();

/*!
 * \brief This pass implements IOS (Inter-Operator-Scheduler) stream schedule policy. It transforms
 * BBNF into ANF and injects stream-related operators (e.g., raf.op.set_stream, raf.op.add_event,
 * and raf.op.wait_event).
 *
 * This pass provides the following config parameters:
 * raf.stream_schedule.ios.block_max_size: tvm::Integer
 *  The maximum block size to schedule.
 * raf.stream_schedule.ios.max_stream_num: tvm::Integer
 *  The maximum number of streams to support.
 * raf.stream_schedule.ios.max_stage_ops: tvm::Integer;
 *  The maximum number of operators in a stage.
 * raf.stream_schedule.ios.search_group_combination: tvm::Bool;
 *  Whether to search the group combination.
 * raf.stream_schedule.ios.warmup: tvm::Integer;
 *  The number of warmups in a measurement.
 * raf.stream_schedule.ios.number: tvm::Integer;
 *  The number of execution times of a repeat.
 * raf.stream_schedule.ios.repeat: tvm::Integer;
 *  The number of repeats in a measurement.
 * raf.stream_schedule.ios.verbose: tvm::Bool;
 *  Whether to print verbose messages.
 * raf.stream_schedule.ios.schedule_units: Array<Array<Op>>;
 *  The schedule units. Each schedule unit is a sequence of operators. IOS would schedule based on
 *  these schedule units.
 *
 * \return The created pass.
 */
Pass IOSStreamSchedule();

/*!
 * \brief Deduplicate a GNF IR (merge the same patterns into function calls).
 * \param forward_steps The additional num of steps to search.
 * \param consider_type Whether considering the type information.
 * \param must_dominate Whether the root node of a subgraph must dominate other nodes in the
 * subgraph.
 * \param salt An optional hash salt.
 * \return The created pass.
 */
Pass Deduplicate(int forward_steps, bool consider_type, bool must_dominate,
                 ir::Optional<ir::String> salt);

/*!
 * \brief This pass works in ANF and group allgather operators for ZeRO.
 * \return The created pass.
 */
Pass GroupAllgather();

// Helper functions

/*!
 * \brief Replace the variables that appear in the args by the bound constant values.
 * \param func The function to mutate.
 * \param args A list of arguments that have bound values.
 * \return The updated experience.
 */
ir::Expr BindParam(ir::Function func, ir::Array<ir::Expr> args);

/*!
 * \brief Infer the type of a given expression.
 * \param expr The expression.
 * \return The expression with checked types.
 */
ir::Expr InferType(ir::Expr expr);

/*!
 * \brief Infer the type of a given func by using values.
 * \param expr The func.
 * \param value The values.
 * \return The func with checked types.
 */
ir::Expr InferTypeWithValues(const ir::Expr& func, const ir::Array<value::Value>& values);

/*!
 * \brief Infer the type of a given expression and IR module.
 * \param expr The expression.
 * \param module The module associated with the expression.
 * \return The expression with checked types.
 */
ir::Expr InferTypeWithModule(const ir::Expr& expr, const ir::IRModule& module);

/*!
 * \brief Eliminate dead code in the give expression
 * \param expr The expression.
 * \return The expression with dead code eliminated
 */
ir::Expr DeadCodeElimination(const ir::Expr& expr);

}  // namespace pass
}  // namespace raf
