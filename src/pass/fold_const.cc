/*!
 * Copyright (c) 2020 by Contributors
 * \file const_fold.cc
 * \brief Folding constants
 */
#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/value.h"
#include "mnm/pass.h"
#include "mnm/executor.h"
#include "mnm/binding.h"

namespace mnm {
namespace pass {
namespace fold_const {

using namespace mnm::ir;
using namespace mnm::op;

class ConstantChecker : private ExprVisitor {
 public:
  // Check whether an expression is constant. The results are memoized.
  bool IsConstant(const Expr& expr) {
    // The `ConstantNode` case is common enough that we check directly for the
    // case here, to avoid the time overhead of dispatching through the vtable
    // and the space overhead of memoizing always-true results.
    if (expr.as<ConstantNode>()) {
      return true;
    }
    const auto it = memo_.find(expr);
    if (it != memo_.end()) return it->second;
    VisitExpr(expr);
    return memo_[expr];  // return memoized result or the default value false
  }

 private:
  std::unordered_map<Expr, bool, ObjectPtrHash, ObjectPtrEqual> memo_;

  void VisitExpr_(const TupleNode* n) final {
    bool result = true;
    for (const auto& field : n->fields) {
      if (!IsConstant(field)) {
        result = false;
        break;
      }
    }
    memo_[GetRef<Tuple>(n)] = result;
  }
};

class ConstantFolder : public ExprMutator {
 public:
  explicit ConstantFolder(IRModule module) : module_(module) {
  }

  Expr VisitExpr_(const LetNode* op) final {
    Expr value = this->Mutate(op->value);
    if (value.as<ConstantNode>()) {
      memo_[op->var] = value;
      return this->Mutate(op->body);
    } else {
      Var var = Downcast<Var>(this->Mutate(op->var));
      Expr body = this->Mutate(op->body);
      if (var.same_as(op->var) && value.same_as(op->value) && body.same_as(op->body)) {
        return GetRef<Expr>(op);
      } else {
        return Let(var, value, body);
      }
    }
  }

  Expr VisitExpr_(const CallNode* call) final {
    static std::unordered_set<std::string> skip_list{"zeros_like", "ones_like", "full_like",
                                                     "full"};

    auto origin_args = call->args;
    Expr res = ExprMutator::VisitExpr_(call);
    call = res.as<CallNode>();

    // We don't constant fold function with zero arguments.
    // This is a heuristic that is useful.
    // For example it is harmful to fold ones(shape=(4, 5)).
    if (call->args.size() == 0) return res;
    const OpNode* op = call->op.as<OpNode>();
    if (op == nullptr) return res;
    if (skip_list.count(op->name)) {
      return res;
    }

    // TODO(haibin): skip stateful ops.
    // TODO(haibin): Evaluate a call to the shape_of operator for tensors with constant shapes.
    // TODO(haibin): Constant evaluation over: alloc_tensor_op, alloc_storage_op
    // shape_func, invoke_tvm_op

    bool all_const_args = true;
    for (Expr arg : call->args) {
      if (!checker_.IsConstant(arg)) {
        all_const_args = false;
      }
    }
    if (all_const_args) {
      return ConstEvaluate(res);
    } else {
      return res;
    }
  }

  Expr VisitExpr_(const TupleGetItemNode* op) final {
    Expr res = ExprMutator::VisitExpr_(op);
    op = res.as<TupleGetItemNode>();
    if (const auto* tuple = op->tuple.as<TupleNode>()) {
      return tuple->fields[op->index];
    } else {
      return res;
    }
  }

 private:
  // Internal constant checker
  ConstantChecker checker_;
  // Module
  IRModule module_;

  // Convert value to expression.
  Expr ObjectToExpr(const ObjectRef& value) {
    if (value->IsInstance<tvm::runtime::NDArray::ContainerType>()) {
      auto nd_array = Downcast<tvm::runtime::NDArray>(value);
      for (auto dim : nd_array.Shape()) {
        CHECK_GT(dim, 0) << "invalid dimension after constant eval";
      }
      return Constant(nd_array);
    } else if (const auto* val = value.as<tvm::runtime::ADTObj>()) {
      tvm::runtime::ADT adt = GetRef<tvm::runtime::ADT>(val);
      Array<ir::Expr> fields;
      for (size_t i = 0; i < adt.size(); ++i) {
        fields.push_back(ObjectToExpr(adt[i]));
      }
      return Tuple(fields);
    } else if (value->IsInstance<value::TensorValueObj>()) {
      return MakeConstant(Downcast<value::TensorValue>(value));
    } else if (value->IsInstance<value::TupleValueObj>()) {
      return MakeConstant(Downcast<value::TupleValue>(value));
    } else {
      LOG(FATAL) << "Cannot handle " << value->GetTypeKey();
      return Expr();
    }
  }

  // Constant evaluate an expression.
  Expr ConstEvaluate(Expr expr) {
    // TODO(haibin): run fuse_op, infer_type passes before execution
    // when these passes are ready
    auto module = GlobalModule();
    return ObjectToExpr(executor::interpreter::Interpret(expr, module));
  }
};

struct BindParamMutator : public ExprMutator {
 public:
  explicit BindParamMutator(const Function func, const Array<ir::Expr>& args) {
    auto* node = func.as<FunctionNode>();
    size_t num_args = args.size();
    for (size_t i = 0; i < num_args; ++i) {
      const Expr& arg = args[i];
      if (const auto* a = arg.as<VarNode>()) {
        if (const auto* bound = binding::LookupBinding(a).as<binding::NDArrayBindingObj>()) {
          // We are only interested in binding parameters that does not require gradient
          // These variables then can be replaced with constant nodes.
          if (!bound->tape.defined()) {
            auto* var = node->params[i].as<VarNode>();
            CHECK(var != nullptr);
            var_map[var] = bound;
          }
        }
      }
    }
  }

  Expr VisitExpr_(const VarNode* node) final {
    if (var_map.find(node) != var_map.end()) {
      return MakeConstant(var_map[node]->value);
    }
    // do nothing and return itself
    return GetRef<Var>(node);
  }

  Expr VisitExpr_(const FunctionNode* fn) final {
    // the input params are not pruned. We only mutate the function body
    Function func = GetRef<Function>(fn);
    auto new_body = Mutate(func->body);
    return Function(func->params, new_body, func->ret_type, func->type_params, func->attrs);
  }

  // var node -> the corresponding ndarray object the var is bound to
  std::unordered_map<const VarNode*, const binding::NDArrayBindingObj*> var_map;
};
}  // namespace fold_const

bool IsConstant(const ir::Expr& e) {
  return fold_const::ConstantChecker().IsConstant(e);
}

ir::Expr FoldConstant(ir::Expr expr, ir::IRModule mod) {
  return fold_const::ConstantFolder(mod).Mutate(expr);
}

ir::Expr BindParam(ir::Function func, ir::Array<ir::Expr> args) {
  return fold_const::BindParamMutator(func, args).Mutate(func);
}

MNM_REGISTER_GLOBAL("mnm.pass_.is_constant").set_body_typed(IsConstant);
MNM_REGISTER_GLOBAL("mnm.pass_.FoldConstant").set_body_typed(FoldConstant);
MNM_REGISTER_GLOBAL("mnm.pass_.BindParam").set_body_typed(BindParam);

}  // namespace pass
}  // namespace mnm
