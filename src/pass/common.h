/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file common.h
 * \brief common utilities
 */
#pragma once

#include <vector>
#include <tvm/ir/type_functor.h>
#include "raf/ir_ext.h"
#include "raf/value.h"
#include "../op/schema/init.h"
#include "../op/schema/memory.h"
#include "../op/schema/transform.h"

using tvm::kType;
using tvm::TypeFunctor;

namespace raf {
namespace pass {

using namespace ir;
using namespace value;

struct ExplicitLetList {
 public:
  std::vector<Var> vars;
  std::vector<Expr> exprs;
  Var ret;

  Expr AsExpr() {
    CHECK_EQ(vars.size(), exprs.size());
    Expr body = ret;
    int n = exprs.size();
    for (int i = n - 1; i >= 0; --i) {
      body = Let(vars[i], exprs[i], body);
    }
    return body;
  }

  void Push(Var var, Expr expr) {
    vars.push_back(var);
    exprs.push_back(expr);
  }

  static std::unique_ptr<ExplicitLetList> make(const Expr& node) {
    std::unique_ptr<ExplicitLetList> ell = std::make_unique<ExplicitLetList>();
    Maker(ell.get()).VisitExpr(node);
    return ell;
  }

  struct Maker : public ExprVisitor {
    explicit Maker(ExplicitLetList* ell) : ell(ell) {
    }

    void VisitExpr_(const LetNode* node) final {
      auto pre_visit = [this](const LetNode* op) {
        ell->vars.push_back(op->var);
        ell->exprs.push_back(op->value);
        const Expr& expr = op->body;
        CHECK(expr->IsInstance<LetNode>() || expr->IsInstance<VarNode>())
            << "ValueError: assumes ANF";
        if (expr->IsInstance<VarNode>()) {
          ell->ret = Downcast<Var>(expr);
        }
      };
      auto post_visit = [](const LetNode* op) {};
      ExpandANormalForm(node, pre_visit, post_visit);
    }
    ExplicitLetList* ell;
  };
};

/*!
 * \brief Cache the compiler_begin annotation op to reduce registry lookup overhead
 * \param void
 * \return compiler_begin op
 */
inline const Op& CompilerBeginOp() {
  static auto op = Op::Get("raf.op.compiler_begin");
  return op;
}

/*!
 * \brief Cache the compiler_end annotation op to reduce registry lookup overhead
 * \param void
 * \return compiler_end op
 */
inline const Op& CompilerEndOp() {
  static auto op = Op::Get("raf.op.compiler_end");
  return op;
}

/*!
 * \brief Remove the compiler_begin/end annotation of the
 * expression.
 * \param expr The input expression to remove annotations from.
 * \param ann_op The specific annotation to remove.
 * \return The expression after remove annotation.
 */
inline Expr RemoveAnnotation(const Expr& expr, const Op& ann_op) {
  const Op& begin_op = CompilerBeginOp();
  const Op& end_op = CompilerEndOp();

  if (ann_op == begin_op) {
    if (expr.as<CallNode>()) {
      const CallNode* call = expr.as<CallNode>();

      // If the CallNode is annotated by compiler_end, then get
      // the args of the compiler_end.
      if (call->op == CompilerEndOp()) {
        CHECK_EQ(call->args.size(), 1U);
        auto input_expr = call->args[0];

        // Remove the compiler_begin annotation of this input_call,
        // and return the expr after annotate it with compiler_end.
        auto new_expr = RemoveAnnotation(input_expr, begin_op);
        Expr ret_expr = Call(call->op, {new_expr}, call->attrs);
        ret_expr->checked_type_ = expr->checked_type_;

        return ret_expr;
      } else if (call->args[0].as<CallNode>() &&
                 call->args[0].as<CallNode>()->op == CompilerBeginOp()) {
        // Remove compiler_begin if exists.
        Array<Expr> new_args;
        for (auto& arg : call->args) {
          const CallNode* arg_call = arg.as<CallNode>();
          CHECK_EQ(arg_call->op, CompilerBeginOp()) << "ValueError: compiler_begin not found";
          CHECK_EQ(arg_call->args.size(), 1U);
          new_args.push_back(arg_call->args[0]);
        }

        Expr new_expr = {Call(call->op, new_args, call->attrs)};
        new_expr->checked_type_ = call->checked_type_;

        return new_expr;
      } else {
        // This expr is not annotated with compiler_begin, return it directly.
        return expr;
      }
    } else if (expr.as<TupleNode>()) {
      // Remove the annotation for TupleNode.
      const TupleNode* tuple = expr.as<TupleNode>();

      // If the fields of the TupleNode is annotated, then remove
      // the annotation, else return this TupleNode directly.
      if (tuple->fields[0].as<CallNode>()->op == CompilerBeginOp()) {
        Array<Expr> new_fields;
        for (auto field : tuple->fields) {
          auto field_call = field.as<CallNode>();
          CHECK_EQ(field_call->op, CompilerBeginOp()) << "ValueError: compiler_begin not found";
          CHECK_EQ(field_call->args.size(), 1U);
          new_fields.push_back(field_call->args[0]);
        }

        Expr new_tuple = {Tuple(new_fields)};
        new_tuple->checked_type_ = expr->checked_type_;

        return new_tuple;
      } else {
        return expr;
      }
    }
  }
  // Remove the compiler_end annotation inside the CallNode.
  else if (ann_op == end_op) {
    if (expr.as<CallNode>()) {
      const CallNode* call = expr.as<CallNode>();
      if (call->op == CompilerEndOp()) {
        // Remove compiler_begin annotations of the input call's arguments.
        return call->args[0];
      } else {
        // If the compiler_end annotation is already removed, then do nothing.
        return expr;
      }
    } else {
      return expr;
    }
  } else {
    LOG(FATAL) << "ValueError: unknown op";
    return Expr();
  }
}

class VarSubstitutor : public MixedModeMutator {
 public:
  explicit VarSubstitutor(const tvm::Map<Var, Var> mapping) : mapping_(mapping) {
  }

  Expr VisitExpr_(const VarNode* var_node) final {
    Var var = GetRef<Var>(var_node);
    if (mapping_.count(var)) {
      return mapping_.at(var);
    }
    return var;
  }

  Expr VisitExpr_(const LetNode* op) override {
    auto pre_visit = [this](const LetNode* op) {
      this->Mutate(op->var);
      this->Mutate(op->value);
    };
    auto post_visit = [this](const LetNode* op) {
      Var var = Downcast<Var>(this->VisitExpr(op->var));
      Expr value = this->Mutate(op->value);
      Expr body = this->Mutate(op->body);

      if (var.same_as(op->var) && value.same_as(op->value) && body.same_as(op->body)) {
        this->memo_[GetRef<Expr>(op)] = GetRef<Expr>(op);
      } else {
        this->memo_[GetRef<Expr>(op)] = Let(var, value, body, op->span);
      }
    };
    ExpandANormalForm(op, pre_visit, post_visit);
    return memo_[GetRef<Expr>(op)];
  }

  Expr Substitute(const Expr& expr) {
    return this->Mutate(expr);
  }

 private:
  tvm::Map<Var, Var> mapping_;
  using MixedModeMutator::VisitExpr_;
};

static inline Function CreateGlobalFunc(const Array<Var>& free_vars, const Expr& body,
                                        Type type_annotation) {
  // As this is a global function, create new vars and set them as free vars for the function
  // using old vars results in malformed-ir because they are not available in this scope.
  Array<Var> new_free_vars;
  tvm::Map<Var, Var> mapping;
  for (auto old_free_var : free_vars) {
    Type type_annotation = old_free_var->checked_type_.defined() ? old_free_var->checked_type()
                                                                 : old_free_var->type_annotation;
    Var new_free_var = MakeVar(old_free_var->name_hint(), type_annotation, {});
    new_free_vars.push_back(new_free_var);
    mapping.Set(old_free_var, new_free_var);
  }
  // The old vars might be used in the body, Substitute them with the new vars
  auto new_body = VarSubstitutor(mapping).Substitute(body);

  // Check that the body is just a var node. If it is, wrap it in a Let node to keep valid ANF form.
  if (auto body_var = new_body.as<VarNode>()) {
    auto new_var = Var(body_var->name_hint(), body_var->type_annotation, {});
    new_body = Let(new_var, new_body, new_var);
  }

  return Function(new_free_vars, new_body, type_annotation, {});
}

// Forward declarations of GetValue functions, which generate Value from
// the given type expression.
Value GetValue(Type type);
Value GetValue(Expr expr);

class TypeGetter : public TypeFunctor<Value(const Type&)> {
  Value VisitType_(const TensorTypeNode* op) {
    return TensorTypeValue::make(GetRef<TensorType>(op));
  }

  Value VisitType_(const TupleTypeNode* op) {
    Array<Value> ret;
    for (const auto& ty : op->fields) {
      ret.push_back(VisitType(ty));
    }
    return TupleValue::make(ret);
  }

  Value VisitType_(const FuncTypeNode* op) {
    // FuncType doesn't really carry value so we return void
    return VoidValue::make();
  }
};

class ValueGetter : public ExprFunctor<Value(const Expr&)> {
  Value VisitExpr_(const RelayConstantNode* op) {
    const ConstantNode* node = static_cast<const ConstantNode*>(op);
    if (const ArrayNode* arr = node->value.as<ArrayNode>()) {
      Array<Value> fields;
      for (const auto& it : *arr) {
        fields.push_back(IntValue::make(DataType::Int(64), (Downcast<IntImm>(it))->value));
      }
      return TupleValue::make(fields);
    }
    return node->value.defined() ? Downcast<Value>(node->value) : NullValue<Value>();
  }

  Value VisitExpr_(const TupleNode* op) {
    Array<Value> values;
    for (auto field : op->fields) {
      values.push_back(VisitExpr(field));
    }
    return TupleValue::make(values);
  }

  Value VisitExpr_(const OpNode* op) {
    return OpValue::make(GetRef<Op>(op));
  }

  Value VisitExpr_(const FunctionNode* op) {
    return ClosureValue::make({}, GetRef<Function>(op));
  }

  Value VisitExprDefault_(const Object* op) {
    const auto* e = static_cast<const ExprNode*>(op);
    return GetValue(e->checked_type());
  }
};

inline Value GetValue(Type type) {
  return TypeGetter()(type);
}

inline Value GetValue(Expr expr) {
  return ValueGetter()(expr);
}

/*!
 * \brief Return the device that the given call node should be on.
 * Note that if the op is *_like(t) (e.g., zeros_like) and t.device != current_device,
 * then this function will return a wrong device (i.e., current_device).
 * However, it should be fine for now as we do not support heterogeneous execution.
 */
inline Device GetOutputDevice(const Call& call) {
#define GET_DEVICE_FROM_SCHEMA(BASE_OP, OP, ARGS, ARG_NAME, DEVICE_ATTR_NAME) \
  {                                                                           \
    static auto target_op = Op::Get(OP);                                      \
    if (BASE_OP == target_op) {                                               \
      Array<Value> arg_values;                                                \
      for (const auto& arg : ARGS) {                                          \
        arg_values.push_back(GetValue(arg));                                  \
      }                                                                       \
      auto schema_args = fschema[BASE_OP](arg_values).as<ARG_NAME>();         \
      CHECK(schema_args != nullptr);                                          \
      return Device((tvm::Device)(*str2dev)(schema_args->DEVICE_ATTR_NAME));  \
    }                                                                         \
  }

  Device device = Device::Current();
  if (auto op_node = call->op.as<OpNode>()) {
    static auto fschema = Op::GetAttrMap<op::FRAFSchema>("FRAFSchema");
    static auto* str2dev = tvm::runtime::Registry::Get("raf._core.core_utils.str2dev");

    Op op = GetRef<Op>(op_node);
    Op base_op = op::IsDialectOp(op) ? op::GetBaseOp(op) : op;
    GET_DEVICE_FROM_SCHEMA(base_op, "raf.op.zeros", call->args, op::schema::InitOpArgs, device);
    GET_DEVICE_FROM_SCHEMA(base_op, "raf.op.ones", call->args, op::schema::InitOpArgs, device);
    GET_DEVICE_FROM_SCHEMA(base_op, "raf.op.full", call->args, op::schema::FullArgs, device);
    GET_DEVICE_FROM_SCHEMA(base_op, "raf.op.arange", call->args, op::schema::ArangeArgs, device);
    GET_DEVICE_FROM_SCHEMA(base_op, "raf.op.one_hot", call->args, op::schema::OneHotArgs, device);
    GET_DEVICE_FROM_SCHEMA(base_op, "raf.op.device_copy", call->args, op::schema::DeviceCopyArgs,
                           dst_device);
  }
  return device;
}

};  // namespace pass
};  // namespace raf
