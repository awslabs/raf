/*!
 * Copyright (c) 2019 by Contributors
 * \file lambda_lift.cc
 * \brief Lambda lift pass
 */
#include <tvm/node/structural_equal.h>
#include <tvm/node/structural_hash.h>
#include <sstream>
#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/pass.h"
#include "./common.h"

namespace mnm {
namespace pass {
namespace lambda_lift {

using namespace mnm::ir;
using namespace mnm::op;

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
      mnm::ir::Var gvar = ir::MakeVar("gvar", {});
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
          letrec_.push_back(op->var);
        }
      }
      auto value = VisitExpr(op->value);
      if (is_lambda) {
        letrec_.pop_back();
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
      auto var = GetRef<Var>(var_node);
      if (!letrec_.empty() && var == letrec_.back()) {
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
      if (!letrec_.empty() && var == letrec_.back()) {
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
        lambda_map_.emplace(letrec_.back(), Call(global, fvs));
      } else {
        lambda_map_.emplace(letrec_.back(), global);
      }
    }
    auto body = Downcast<Function>(ExprMutator::VisitExpr_(func_node));

    // When performing this optimization there are two cases.
    //
    // The first case in which we have no free variables
    // we can just lift the function into the global
    // environment without needing to allocate a closure.
    //
    // The second case requires that we generate a special
    // function which makes a distinction between allocating
    // a closure, and then the code for the closure.
    //
    // We represent a closure allocation by lifting the
    // closure to a global function which takes its
    // captured arguments and then directly returns
    // the function representing the closure's code.
    //
    // When we generate code later on a call to the "outer"
    // function marked as a closure is used to emit allocation
    // code for the closure's environment.
    //
    // The "inner" function should be used to generate the
    // code for the closure.

    Function lifted_func;
    if (captured_vars.size() == 0) {
      lifted_func = Function(body->params, body->body, {}, {});
    } else {
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
      // If we need to allocate a closure,
      // we pass the variables in its environment here.
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
  // initialized in constructor
  ir::IRModule module_;
  std::unordered_map<Var, Expr, ObjectPtrHash, ObjectPtrEqual> lambda_map_;
  std::vector<Var> letrec_;
};

}  // namespace lambda_lift

Pass LambdaLift() {
  return CreateModulePass(
      [=](IRModule mod, const PassContext& pass_ctx) {
        return lambda_lift::LambdaLifter(mod).Lift();
      },
      0, "LambdaLift", {});
}

MNM_REGISTER_GLOBAL("mnm.pass_.LambdaLift").set_body_typed(LambdaLift);

}  // namespace pass
}  // namespace mnm
