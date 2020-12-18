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
  return WithAttr(std::move(func), tvm::relay::attr::kClosure, tvm::Integer(1));
}

class LambdaLifter : public ExprMutator {
 public:
  explicit LambdaLifter(Module module) : module_(module) {
  }

  Expr VisitExpr_(const LetNode* let_node) final {
    bool is_lambda = false;
    if (auto func = let_node->value.as<FunctionNode>()) {
      if (!func->HasNonzeroAttr(tvm::relay::attr::kPrimitive)) {
        is_lambda = true;
        letrec_.push_back(let_node->var);
      }
    }
    auto value = VisitExpr(let_node->value);
    if (is_lambda) {
      letrec_.pop_back();
    }
    auto body = VisitExpr(let_node->body);
    return Let(let_node->var, value, body);
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
    if (func->HasNonzeroAttr(tvm::relay::attr::kPrimitive)) {
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
      lifted_func = Function(captured_vars, body, func->func_type_annotation(), {});
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

  ir::Module Lift() {
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
  Module module_;
  std::unordered_map<Var, Expr, ObjectPtrHash, ObjectPtrEqual> lambda_map_;
  std::vector<Var> letrec_;
};

}  // namespace lambda_lift

ir::Module LambdaLift(ir::Module mod) {
  return lambda_lift::LambdaLifter(mod).Lift();
}

MNM_REGISTER_GLOBAL("mnm.pass_.LambdaLift").set_body_typed(LambdaLift);

}  // namespace pass
}  // namespace mnm
