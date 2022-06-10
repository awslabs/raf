/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file annotate_collective_ops.cc
 * \brief Add memory copy ops to pipeline memory copies in multi-tensor collective ops.
 */
#include "raf/device.h"
#include "raf/ir.h"
#include "raf/ir_ext.h"
#include "raf/dist_config.h"
#include "raf/pass.h"
#include "raf/analysis.h"
#include "raf/op.h"
#include "raf/op_utils.h"
#include "./common.h"
#include "raf/stream_pool.h"
#include "memory_op_utils.h"

namespace raf {
namespace pass {
namespace annotate_collective_ops {
using namespace raf::ir;
using namespace raf::op;
using namespace raf::value;
using namespace raf::op::schema;
using namespace raf::analysis;
using namespace raf::pass::memory_op_utils;

template <typename T>
using ExprMap = std::unordered_map<Expr, T, ObjectPtrHash, ObjectPtrEqual>;

class CollectiveOpAnalyzer : ExprVisitor {
 public:
  bool Analyse(const Expr& expr) {
    VisitExpr(expr);
    if (fuse_tensor_before_op.empty() && defuse_tensor_after_op.empty()) {
      return false;
    }
    return true;
  }

  void VisitExpr_(const CallNode* call) {
    auto call_expr = GetRef<Expr>(call);
    if (IsCollectiveOp(call->op)) {
      UpdateOpToAddInfo_(call);
    }
  }

  void VisitExpr_(const FunctionNode* op) {
    // currently assumes closures do not contain collectives
    auto ell = ExplicitLetList::make(op->body);
    for (auto& expr : ell->exprs) {
      CHECK(!(expr.as<CallNode>() && IsCollectiveOp(expr.as<CallNode>()->op)))
          << "Unimplemented: Collectives in closures are currently not supported.";
    }
  }

  void VisitExpr_(const LetNode* op) {
    expr_idx_map[op->value] = current_idx_;
    this->VisitExpr(op->value);
    current_idx_++;
    this->VisitExpr(op->body);
  }

  ExprMap<int> expr_idx_map;

  // the set of ops which needs a fuse_tensor before it
  std::unordered_set<int> fuse_tensor_before_op;

  // the set of ops which needs a defuse_tensor after it
  std::unordered_set<int> defuse_tensor_after_op;
  // arguments of defuse_tensor
  std::unordered_map<int, DefuseTensorArgs> defuse_tensor_args;

 private:
  void UpdateOpToAddInfo_(const CallNode* call) {
    auto op_to_add = GetFuseAndDefuseOpToAdd(call);
    FuseOp fuse_op_to_add = op_to_add.first;
    DefuseOp defuse_op_to_add = op_to_add.second;
    if (fuse_op_to_add == FuseOp::kFuse) {
      fuse_tensor_before_op.emplace(current_idx_);
    }
    if (defuse_op_to_add == DefuseOp::kDefuse) {
      defuse_tensor_after_op.emplace(current_idx_);
      defuse_tensor_args.emplace(current_idx_, GetDefuseTensorArgs(call));
    }
  }

 private:
  // current_idx_ keeps track of current "depth" (or the index of the op we are visiting in the ANF
  // order) when expanding the nested let exprs
  int current_idx_ = 0;
};

class CollectiveOpAnnotator : ExprVisitor {
 public:
  explicit CollectiveOpAnnotator(const FunctionNode* func, bool use_memory_copy_ops)
      : func_(func), use_memory_copy_ops_(use_memory_copy_ops) {
  }

  void VisitExpr_(const LetNode* op) {
    Var orig_var = op->var;
    Expr orig_value = op->value;
    int orig_value_idx = analyzer_.expr_idx_map[orig_value];

    // if visiting collective ops, decide whether to add ops before and after them
    if (orig_value->IsInstance<CallNode>() && IsCollectiveOp(orig_value.as<CallNode>()->op)) {
      Var substituted_var;
      bool is_orig_var_substituted = false;
      Expr substituded_value;
      bool is_orig_value_substituted = false;

      // add fuse_tensor before collective op
      if (analyzer_.fuse_tensor_before_op.count(orig_value_idx)) {
        Var orig_value_var = GetRef<Var>(orig_value.as<CallNode>()->args[0].as<VarNode>());
        std::string fused_name_hint = "fused_";
        fused_name_hint += std::string(orig_value_var->name_hint());
        Var fused_orig_value_var = raf::ir::MakeVar(fused_name_hint, {});
        Expr fuse_tensor_value = CreateFuseTensorOp(orig_value_var);
        ell_->Push(fused_orig_value_var, fuse_tensor_value);

        std::string fused_tuple_name_hint = "fused_tuple_";
        fused_tuple_name_hint += std::string(orig_value_var->name_hint());
        Var fused_tuple_orig_value_var = raf::ir::MakeVar(fused_tuple_name_hint, {});
        ell_->Push(fused_tuple_orig_value_var, Tuple({fused_orig_value_var}));

        Array<Expr> args;
        args.push_back(fused_tuple_orig_value_var);
        for (int i = 1; i < orig_value.as<CallNode>()->args.size(); ++i) {
          args.push_back(orig_value.as<CallNode>()->args[i]);
        }
        substituded_value = Call(orig_value.as<CallNode>()->op, args);
        is_orig_value_substituted = true;
      }

      // add var and expr being visited, which may have been substituted
      if (analyzer_.defuse_tensor_after_op.count(orig_value_idx)) {
        std::string to_defuse_name_hint = std::string(orig_var->name_hint()) + "_to_defuse";
        substituted_var = raf::ir::MakeVar(to_defuse_name_hint, {});
        is_orig_var_substituted = true;
      }
      ell_->Push(is_orig_var_substituted ? substituted_var : orig_var,
                 is_orig_value_substituted ? substituded_value : orig_value);

      // add defuse_tensor after collective op
      if (analyzer_.defuse_tensor_after_op.count(orig_value_idx)) {
        DefuseTensorArgs args = analyzer_.defuse_tensor_args[orig_value_idx];
        Expr defuse_value = CreateDefuseTensorOp(substituted_var, args);
        ell_->Push(orig_var, defuse_value);
      }
    } else {
      ell_->Push(orig_var, orig_value);
    }

    ell_->ret = ell_->vars.back();
    VisitExpr(op->body);
  }

  Function Run() {
    if (!use_memory_copy_ops_ || !analyzer_.Analyse(func_->body)) {
      // do nothing.
      return GetRef<Function>(func_);
    }

    ell_ = std::make_unique<ExplicitLetList>();
    VisitExpr(func_->body);

    return Function(func_->params, ell_->AsExpr(), {}, {});
  }

 protected:
  Expr CreateFuseTensorOp(Expr arg) {
    static Op op = Op::Get("raf.op.fuse_tensor");
    Array<Expr> args({arg});
    return Call(op, args);
  }

  Expr CreateDefuseTensorOp(Expr arg, const DefuseTensorArgs& args) {
    static Op op = Op::Get("raf.op.defuse_tensor");
    Expr sizes = MakeConstant(ArrayToIntTuple(std::get<0>(args)));
    Expr shapes = MakeConstant(ArrayToIntTuple(std::get<1>(args)));
    Expr shape_indices = MakeConstant(ArrayToIntTuple(std::get<2>(args)));
    Array<Expr> args_({arg, sizes, shapes, shape_indices});
    return Call(op, args_);
  }

 private:
  const FunctionNode* func_;
  bool use_memory_copy_ops_ = false;
  CollectiveOpAnalyzer analyzer_;
  std::unique_ptr<ExplicitLetList> ell_;
};

}  // namespace annotate_collective_ops

TVM_REGISTER_PASS_CONFIG_OPTION("raf.annotate_collective_ops.use_memory_copy_ops", Bool);

Pass AnnotateCollectiveOps() {
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    bool use_memory_copy_ops = static_cast<bool>(
        pc->GetConfig("raf.annotate_collective_ops.use_memory_copy_ops", Bool(false)).value());
    return annotate_collective_ops::CollectiveOpAnnotator(f.operator->(), use_memory_copy_ops)
        .Run();
  };
  auto func_pass = CreateRAFFunctionPass(pass_func, 0, "AnnotateCollectiveOps", {});
  PassInfo pass_info(0, "AnnotateCollectiveOps", {});
  return RAFSequential({InferType(), func_pass, EraseType()}, pass_info);
}

RAF_REGISTER_GLOBAL("raf.pass_.AnnotateCollectiveOps").set_body_typed(AnnotateCollectiveOps);

}  // namespace pass
}  // namespace raf
