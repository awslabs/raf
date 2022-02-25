/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file inline_primitives.cc
 * \brief Ensure that primitives only appear in the call position.
 */
#include "raf/ir.h"
#include "raf/ir_ext.h"
#include "raf/pass.h"

namespace raf {
namespace pass {
namespace inline_primitives {

using namespace raf::ir;

/* This pass will eliminate primitives which have been lifted by the ANF
 * transform inlining them directly into call sites.
 *
 * This makes VM related code generation easier as the call target is always
 * a primitive function.
 *
 * let prim = fn(...) { ... };
 * prim(...)
 *
 * will become:
 *
 * (fn(...) { ... })(...)
 */
class PrimitiveInliner : public ExprMutator {
 public:
  explicit PrimitiveInliner(const IRModule& module) : module_(module) {
  }

  Expr VisitExpr_(const LetNode* let_node) {
    auto pre_visit = [this](const LetNode* op) {
      var_map.insert({op->var, this->VisitExpr(op->value)});
    };
    auto post_visit = [this](const LetNode* op) {
      // Rely on the Memoizer to cache pre-visit values
      Expr value = this->VisitExpr(op->value);
      // Visit body and cache the op
      Expr body = this->VisitExpr(op->body);
      auto expr = GetRef<Expr>(op);
      this->memo_[expr] = Let(op->var, value, body);
    };
    ExpandANormalForm(let_node, pre_visit, post_visit);
    return memo_[GetRef<Expr>(let_node)];
  }

  Expr VisitExpr_(const CallNode* call) {
    Expr op = call->op;
    // For now just collapse the chain of variables to see if
    // they point to a primitive function.
    const VarNode* var_node;

    // Collapse a chain of let bindings
    //
    // let x = fn (..) { .. };
    // let y = x
    // let w = y
    // in w(...)
    while ((var_node = op.as<VarNode>())) {
      auto var = GetRef<Var>(var_node);
      DLOG(INFO) << "Var: " << var << std::endl;
      auto it = var_map.find(GetRef<Var>(var_node));
      if (it != var_map.end()) {
        op = it->second;
      } else {
        return ExprMutator::VisitExpr_(call);
      }
    }

    if (auto func = op.as<FunctionNode>()) {
      if (func->HasNonzeroAttr(attr::kPrimitive)) {
        tvm::Array<Expr> call_args;
        for (auto arg : call->args) {
          auto new_arg = VisitExpr(arg);
          call_args.push_back(new_arg);
        }
        return Call(GetRef<Function>(func), call_args, call->attrs, call->type_args);
      }
    }

    if (auto global = op.as<GlobalVarNode>()) {
      tvm::Array<Expr> call_args;
      for (auto arg : call->args) {
        auto new_arg = VisitExpr(arg);
        call_args.push_back(new_arg);
      }
      return Call(GetRef<GlobalVar>(global), call_args, call->attrs, call->type_args);
    }

    return ExprMutator::VisitExpr_(call);
  }

  Expr VisitExpr_(const FunctionNode* func) {
    if (func->HasNonzeroAttr(attr::kPrimitive)) {
      return GetRef<Function>(func);
    } else {
      return ExprMutator::VisitExpr_(func);
    }
  }

  IRModule Inline() {
    auto gvar_funcs = module_->functions;
    for (auto pair : gvar_funcs) {
      auto global = pair.first;
      auto base_func = pair.second;
      if (auto* n = base_func.as<FunctionNode>()) {
        if (n->GetAttr<String>(attr::kCompiler).defined()) continue;
        auto func = GetRef<Function>(n);

        DLOG(INFO) << "Before inlining primitives: " << global << std::endl
                   << ir::AsText(func, false);

        func = Function(func->params, VisitExpr(func->body), func->ret_type, func->type_params,
                        func->attrs);
        module_->Add(global, func, true);

        DLOG(INFO) << "After inlining primitives: " << global << std::endl
                   << ir::AsText(func, false);
      }
    }
    return module_;
  }

 private:
  IRModule module_;
  std::unordered_map<Var, Expr, ObjectPtrHash, ObjectPtrEqual> var_map;
};

}  // namespace inline_primitives

Pass InlinePrimitives() {
  TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func = [=](IRModule m, PassContext pc) {
    return inline_primitives::PrimitiveInliner(m).Inline();
  };
  auto inline_pass = CreateModulePass(pass_func, 0, "Inline", {});
  // Eliminate dead code for each function after inlining.
  return RAFSequential({inline_pass, DeadCodeElimination()}, "InlinePrimitives");
}

RAF_REGISTER_GLOBAL("raf.pass_.InlinePrimitives").set_body_typed(InlinePrimitives);

}  // namespace pass
}  // namespace raf
