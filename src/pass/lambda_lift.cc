/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file lambda_lift.cc
 * \brief Lambda lift pass
 */
#include <tvm/node/structural_equal.h>
#include <tvm/node/structural_hash.h>
#include <sstream>
#include "raf/op.h"
#include "raf/ir.h"
#include "raf/pass.h"
#include "./common.h"

namespace raf {
namespace pass {
namespace lambda_lift {

using namespace raf::ir;
using namespace raf::op;

inline std::string GenerateName(const Function& func) {
  size_t hash = tvm::StructuralHash()(func);
  return std::string("lifted_name") + std::to_string(hash);
}

Function MarkClosure(Function func) {
  return WithAttr(std::move(func), attr::kClosure, tvm::Integer(1));
}

Expr ANFNormalizer(const Let& let) {
  const LetNode* let_node = let.get();

  // Simplifies
  // let %a2 = %a3 = @lifted_name4372026607701689919(%y);
  // %a3
  // to simply let %a2 = @lifted_name4372026607701689919(%y);
  if (auto nested_let_node = let_node->value.as<LetNode>()) {
    if (tvm::StructuralEqual()(nested_let_node->var, nested_let_node->body)) {
      if (auto call = nested_let_node->value.as<CallNode>()) {
        if (call->op.as<GlobalVarNode>()) {
          return Let(let_node->var, nested_let_node->value, let_node->body);
        }
      }
    }
  }

  // Lambda lift can lead to scenarios where the lifted global call is not present in let binding
  // The following code finds such scenarios
  // free_var %a, %b, free_var %y;
  // %0 = @lifted_name3932855059181273852(%y); --> This is not let binding
  // let %a13 = %0(%a, %b);
  // %a13
  //
  // This above gets simplified to
  //    let %gvar = @lifted_name4372026607701689919(%y);
  //    let %a13 = %gvar(%a, %b);
  if (auto value_call_node = let_node->value.as<CallNode>()) {
    if (auto op_call_node = value_call_node->op.as<CallNode>()) {
      raf::ir::Var gvar = ir::MakeVar("gvar", {});
      auto new_let = Let(
          gvar, GetRef<Call>(op_call_node),
          Let(let_node->var,
              Call(gvar, value_call_node->args, value_call_node->attrs, value_call_node->type_args),
              let_node->body));
      return new_let;
    }
  }
  return let;
}

class LambdaLifter : public ExprMutator {
 public:
  explicit LambdaLifter(ir::IRModule module) : module_(module) {
  }

  Expr VisitExpr_(const LetNode* let_node) final {
    auto pre_visit = [this](const LetNode* op) {
      bool is_lambda = false;
      if (auto func = op->value.as<FunctionNode>()) {
        if (!func->HasNonzeroAttr(tvm::relay::attr::kPrimitive)) {
          is_lambda = true;
          curr_lambda_lets_.push_back(op->var);
        }
      }
      auto value = VisitExpr(op->value);
      if (is_lambda) {
        curr_lambda_lets_.pop_back();
      }
    };
    auto post_visit = [this](const LetNode* op) {
      auto expr = GetRef<Expr>(op);
      auto value = VisitExpr(op->value);
      auto body = VisitExpr(op->body);
      auto new_let = Let(op->var, value, body);
      this->memo_[expr] = ANFNormalizer(new_let);
    };
    ExpandANormalForm(let_node, pre_visit, post_visit);
    return memo_[GetRef<Expr>(let_node)];
  }

  Expr VisitExpr_(const CallNode* call_node) final {
    auto call = Downcast<Call>(ExprMutator::VisitExpr_(call_node));
    if (auto var_node = call_node->op.as<VarNode>()) {
      // If this is calling a lambda function, which should have been lifted,
      // then we update the callee to the global var of the lifted function.
      auto var = GetRef<Var>(var_node);
      if (!curr_lambda_lets_.empty() && var == curr_lambda_lets_.back()) {
        auto it = lambda_map_.find(var);
        CHECK(it != lambda_map_.end());
        return Call(it->second, call->args, call_node->attrs, call_node->type_args);
      }
    }
    return std::move(call);
  }

  Expr VisitExpr_(const FunctionNode* func_node) final {
    auto func = GetRef<Function>(func_node);

    // We should not transform primitive functions.
    if (func->HasNonzeroAttr(attr::kPrimitive)) {
      return std::move(func);
    }

    auto name = GenerateName(func);
    auto global = GlobalVar(name);
    auto free_vars = FreeVars(func);

    Array<Var> captured_vars;
    bool recursive = false;
    for (const auto& var : free_vars) {
      if (!curr_lambda_lets_.empty() && var == curr_lambda_lets_.back()) {
        recursive = true;
        continue;
      }
      captured_vars.push_back(var);
    }
    if (recursive) {
      if (!captured_vars.empty()) {
        Array<Expr> fvs;
        for (auto fv : captured_vars) {
          fvs.push_back(fv);
        }
        lambda_map_.emplace(curr_lambda_lets_.back(), Call(global, fvs));
      } else {
        lambda_map_.emplace(curr_lambda_lets_.back(), global);
      }
    }
    auto body = Downcast<Function>(ExprMutator::VisitExpr_(func_node));

    // Lift the closure to be a global function.
    Function lifted_func;
    if (captured_vars.size() == 0) {
      // The function is self-contained. We can just lift it without allocating a closure.
      lifted_func = Function(body->params, body->body, {}, {});
    } else {
      // The function has free variables, meaning that the function body uses variables
      // other then the function arguments, so we allocate a closure to preserve the semantics.
      // Example:
      // def @lifted_name17350894363744824019(%y, Closure=1) {
      //   fn (%x: Tensor[(1, 100), float32]) {
      //     ... %x ... %y ...
      //   }
      // }
      // def @main(%x, %y) {
      //   let %a1 = @lifted_name17350894363744824019(%y);
      //   let %a2 = %a1(%x);
      //   %a2
      // }
      // To flat the above function, apply FlattenClosure after this pass.
      lifted_func = CreateGlobalFunc(captured_vars, body, func->func_type_annotation());
      lifted_func = MarkClosure(lifted_func);
    }

    CHECK(lifted_func.defined());

    if (module_->ContainGlobalVar(name)) {
      const auto existing_func = module_->Lookup(name);
      CHECK(tvm::StructuralEqual()(lifted_func, existing_func)) << "lifted function hash collision";
      // If an identical function already exists, use its global var.
      global = module_->GetGlobalVar(name);
    } else {
      // Add the lifted function to the module.
      module_->Add(global, lifted_func);
    }

    if (captured_vars.size() == 0) {
      return std::move(global);
    } else {
      // If we need to allocate a closure, we pass the variables in its environment here.
      Array<Expr> fvs;
      for (auto fv : captured_vars) {
        fvs.push_back(fv);
      }
      return Call(global, fvs);
    }
  }

  ir::IRModule Lift() {
    auto glob_funcs = module_->functions;
    for (auto pair : glob_funcs) {
      if (auto* n = pair.second.as<FunctionNode>()) {
        auto func = GetRef<Function>(n);
        func = Function(func->params, VisitExpr(func->body), func->ret_type, func->type_params,
                        func->attrs);
        module_->Add(pair.first, func, true);
      }
    }
    return module_;
  }

 private:
  /*! \brief The working module. */
  ir::IRModule module_;
  /*! \brief Mapping from the let-binding var of a closure to its lifted global var. */
  std::unordered_map<Var, Expr, ObjectPtrHash, ObjectPtrEqual> lambda_map_;
  /*! \brief A list of current visiting lambda vars. Recursion happens when list size > 1. */
  std::vector<Var> curr_lambda_lets_;
};

}  // namespace lambda_lift

Pass LambdaLift() {
  return CreateModulePass(
      [=](IRModule mod, const PassContext& pass_ctx) {
        return lambda_lift::LambdaLifter(mod).Lift();
      },
      0, "LambdaLift", {});
}

RAF_REGISTER_GLOBAL("raf.pass_.LambdaLift").set_body_typed(LambdaLift);

}  // namespace pass
}  // namespace raf
