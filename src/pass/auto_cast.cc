/*!
 * Copyright (c) 2020 by Contributors
 * \file auto_cast.cc
 * \brief AutoCast pass
 */
#include <tvm/ir/transform.h>

#include <stack>
#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/value.h"
#include "mnm/pass.h"
#include "./let_list.h"
#include "./common.h"

namespace mnm {
namespace pass {
namespace auto_cast {

using namespace mnm::ir;
using namespace mnm::op;
using namespace tvm;
using namespace runtime;
using namespace mnm::value;

using TypeHint = Type;
using TypeHints = Array<TypeHint>;

/*! \brief Try to cast TypeHint to PrimType. */
inline PrimType TryToPrimType(TypeHint type) {
  auto prim_type = type.as<PrimTypeNode>();
  CHECK(prim_type != nullptr) << "Expected PrimType as the type hint, but got "
                              << type->GetTypeKey();
  return GetRef<PrimType>(prim_type);
}

/*! \brief Try to cast TypeHint to TupleType. */
inline TupleType TryToTupleType(TypeHint type) {
  auto tuple_type = type.as<TupleTypeNode>();
  CHECK(tuple_type != nullptr) << "Expected TupleType as the type hint, but got "
                               << type->GetTypeKey();
  return GetRef<TupleType>(tuple_type);
}

/*! \brief Whether the given type hint means skip. */
inline bool SkipTypeHint(TypeHint type) {
  auto prim_type = type.as<PrimTypeNode>();
  return (prim_type && GetRef<PrimType>(prim_type)->dtype.is_void());
}

/*! \brief Replace TypeVar("amp") in the type hints with the target_dtype. */
TypeHints ReplaceTypeVarWithAMPType(const TypeHints& type_hints, const DataType target_dtype) {
  TypeHints new_type_hints;
  for (auto type_hint : type_hints) {
    if (auto tuple_type = type_hint.as<TupleTypeNode>()) {
      TypeHints field_type_hints;
      for (auto field_type_hint : tuple_type->fields) {
        field_type_hints.push_back(ReplaceTypeVarWithAMPType({field_type_hint}, target_dtype)[0]);
      }
      new_type_hints.push_back(TupleType(field_type_hints));
    } else {
      const auto* type_var = type_hint.as<TypeVarNode>();
      if (type_var) {
        // Replace TypeVar("amp") with the target dtype.
        CHECK_EQ(type_var->name_hint, "amp");
        new_type_hints.push_back(PrimType(target_dtype));
      } else {
        // Otherwise follow the given type hint.
        CHECK(type_hint->IsInstance<PrimTypeNode>())
            << "Expected prim type hint, but got " << type_hint->GetTypeKey();
        new_type_hints.push_back(type_hint);
      }
    }
  }
  return new_type_hints;
}

/*! \brief A helper function that generates the type hint from the given type.
 * The tensor is hinted to be TypeVar("amp") so that it will be replaced with
 * the user-specified AMP dtype later.
 */
TypeHint GetDefaultTypeHintHelper(const Type& type) {
  if (auto tuple_type = type.as<TupleTypeNode>()) {
    TypeHints field_types;
    for (auto field_type : tuple_type->fields) {
      field_types.push_back(GetDefaultTypeHintHelper(field_type));
    }
    return TupleType(field_types);
  }
  CHECK(type->IsInstance<TensorTypeNode>())
      << "Expected tensor type node, but got " << type->GetTypeKey();
  return TypeVar("amp", TypeKind::kType);
}

/*! \brief Generate default type hints for an expression given the arguments and return type.
 */
TypeHints GenTypeHints(const Array<Expr>& args, const Type& ret_type, const DataType target_dtype,
                       const Expr op_node = Expr()) {
  TypeHints type_hints;

  // Use the custom casting type hint if available.
  static auto frule = Op::GetAttrMap<op::FMNMCastRule>("FMNMCastRule");
  if (op_node.defined() && op_node.as<OpNode>() != nullptr) {
    const Op op = Downcast<Op>(op_node);
    if (frule.count(op)) {
      type_hints = frule[op](args, ret_type);
    }
  }

  // When no custom type hints, default all tensor argument type hints to be in the AMP dtype.
  if (type_hints.size() == 0) {
    for (auto arg : args) {
      if (arg->IsInstance<ConstantNode>()) {
        type_hints.push_back(PrimType(DataType::Void()));
      } else {
        type_hints.push_back(GetDefaultTypeHintHelper(arg->checked_type()));
      }
    }
    type_hints.push_back(GetDefaultTypeHintHelper(ret_type));  // Return type.
  }

  return ReplaceTypeVarWithAMPType(type_hints, target_dtype);
}

/*! \brief Generate a cast call that casts the given expr to the target dtype. */
inline Expr GenCastCall(Expr expr, DataType dtype) {
  static const Op& op = Op::Get("mnm.op.cast");
  const auto old_type = Downcast<TensorType>(expr->checked_type());
  std::string target_dtype;
  if (dtype.is_float16()) {
    target_dtype = "float16";
  } else {
    target_dtype = "float32";
  }
  auto cast_call = Call(op, {expr, MakeConstant(StringValue::make(target_dtype))}, {});
  DataType new_dtype = DataType(ir::String2DLDataType(target_dtype));
  cast_call->checked_type_ = TensorType(old_type->shape, new_dtype);
  return cast_call;
}

/*! \brief Given a type and update its dtype based on the given type hint. */
Type UpdateDTypeByHint(const Type ori_type, const TypeHint& target_type) {
  if (ori_type->IsInstance<TensorTypeNode>()) {
    auto old_type = Downcast<TensorType>(ori_type);
    return TensorType(old_type->shape, TryToPrimType(target_type)->dtype);
  } else if (ori_type->IsInstance<TupleTypeNode>()) {
    auto old_tuple_type = Downcast<TupleType>(ori_type);
    auto target_tuple_type_fields = TryToTupleType(target_type)->fields;
    CHECK_EQ(target_tuple_type_fields.size(), old_tuple_type->fields.size());

    Array<Type> new_field_types;
    for (size_t i = 0; i < old_tuple_type->fields.size(); ++i) {
      auto field_type = old_tuple_type->fields[i];
      new_field_types.push_back(UpdateDTypeByHint(field_type, target_tuple_type_fields[i]));
    }
    return TupleType(new_field_types);
  }
  LOG(FATAL) << "Unsupported type: " << ori_type->GetTypeKey();
}

class AutoCastMutator : public ExprMutator {
 public:
  AutoCastMutator(const String amp_dtype, const String out_dtype) : scopes_{LetList()} {
    amp_dtype_ = DataType(ir::String2DLDataType(amp_dtype));
    out_dtype_ = DataType(ir::String2DLDataType(out_dtype));
  }

  Expr VisitExpr_(const LetNode* node) {
    scopes_.emplace_back();
    auto& scope = scopes_.back();
    Expr body;
    do {
      auto new_value = VisitExpr(node->value);
      node->var->checked_type_ = new_value->checked_type();
      scope.Push(node->var, new_value);
      let_vars_.emplace(node->var, new_value);
      body = node->body;
      node = body.as<LetNode>();
    } while (node);
    auto new_body = VisitExpr(body);

    // Cast output tensors if needed.
    TypeHint ret_type_hint = GenTypeHints({new_body}, new_body->checked_type(), out_dtype_)[0];
    new_body = CastExpr(new_body, ret_type_hint);
    auto ret = scopes_.back().Get(new_body);
    ret->checked_type_ = new_body->checked_type();
    scopes_.pop_back();
    return ret;
  }

  Expr VisitExpr_(const CallNode* node) {
    const auto* old_type = node->checked_type().as<TensorTypeNode>();
    TypeHints type_hints = GenTypeHints(node->args, node->checked_type(), amp_dtype_, node->op);

    Array<Expr> call_args;
    auto op = Downcast<Op>(node->op);
    CHECK_EQ(type_hints.size(), node->args.size() + 1)
        << "Type hint number and argument size of " << op->name << " are mismatching";
    for (size_t i = 0; i < node->args.size(); ++i) {
      auto arg = VisitExpr(node->args[i]);
      if (SkipTypeHint(type_hints[i])) {
        call_args.push_back(arg);
      } else {
        call_args.push_back(CastExpr(arg, type_hints[i]));
      }
    }

    auto new_call = Call(node->op, call_args, node->attrs, node->type_args);
    new_call->checked_type_ = UpdateDTypeByHint(node->checked_type(), type_hints.back());
    return new_call;
  }

  Expr VisitExpr_(const TupleNode* node) {
    Array<Expr> fields;
    Array<Type> types;
    for (const auto& e : node->fields) {
      auto f = VisitExpr(e);
      fields.push_back(f);
      types.push_back(f->checked_type());
    }
    Tuple ret(fields);
    ret->checked_type_ = TupleType(types);
    return ret;
  }

  Expr VisitExpr_(const TupleGetItemNode* node) {
    auto tup = VisitExpr(node->tuple);
    TupleGetItem ret(tup, node->index);
    ret->checked_type_ = Downcast<TupleType>(tup->checked_type())->fields[node->index];
    return ret;
  }

  Expr VisitExpr_(const IfNode* node) override {
    Expr cond = VisitExpr(node->cond);
    Expr true_branch = VisitExpr(node->true_branch);
    Expr false_branch = VisitExpr(node->false_branch);
    Expr ret = If(cond, true_branch, false_branch);
    // TODO(comaniac): Propagate IfNode type.
    return ret;
  }

  Expr VisitExpr_(const FunctionNode* node) {
    if (node->HasNonzeroAttr(attr::kPrimitive)) {
      return GetRef<Expr>(node);
    }

    Array<Var> params;
    Array<Type> param_types;
    for (const auto& p : node->params) {
      Var param = Downcast<Var>(VisitExpr(p));
      params.push_back(param);
      param_types.push_back(param->checked_type());
    }
    Expr body = VisitExpr(node->body);
    Type ret_type = body->checked_type();
    Function func(params, body, ret_type, node->type_params, node->attrs);
    func->checked_type_ = FuncType(param_types, ret_type, node->type_params, {});
    return func;
  }

 private:
  /*! \brief Generate a tuple of casted tensors. */
  Expr CastTuple(const Expr arg, const TupleType& tuple_type_hint) {
    auto& scope = scopes_.back();

    // Find the root tuple.
    auto expr = let_vars_[Downcast<Var>(arg)];
    while (!expr->IsInstance<TupleNode>()) {
      // If the expr is in tuple type but not a tuple node, then it means the expr
      // is a TupleGetItem that points to another tuple node, so we need to
      // trace back to find the root tuple node to obtain its real type.
      CHECK(expr->IsInstance<TupleGetItemNode>());
      auto tgi = Downcast<TupleGetItem>(expr);
      expr = let_vars_[Downcast<Var>(tgi->tuple)];
      expr = Downcast<Tuple>(expr)->fields[tgi->index];
      expr = let_vars_[Downcast<Var>(expr)];
    }
    auto tuple = Downcast<Tuple>(expr);
    CHECK_EQ(tuple->fields.size(), tuple_type_hint->fields.size());

    Array<Expr> fields;
    for (size_t i = 0; i < tuple->fields.size(); ++i) {
      auto field = tuple->fields[i];
      if (field->IsInstance<ConstantNode>()) {
        fields.push_back(field);
      } else {
        auto field_type = field->checked_type();
        Expr new_expr;
        if (field_type->IsInstance<TensorTypeNode>()) {
          new_expr = CastExpr(field, tuple_type_hint->fields[i]);
        } else if (field_type->IsInstance<TupleTypeNode>()) {
          auto tuple_type = TryToTupleType(tuple_type_hint->fields[i]);
          new_expr = CastTuple(field, tuple_type);
        } else {
          LOG(FATAL) << "Unsupported field type: " << field_type->GetTypeKey();
        }
        auto new_var = scope.Push(new_expr);
        new_var->checked_type_ = new_expr->checked_type_;
        fields.push_back(new_var);
      }
    }
    auto new_tuple = Tuple(fields);
    new_tuple->checked_type_ = UpdateDTypeByHint(tuple->checked_type(), tuple_type_hint);
    return new_tuple;
  }

  /*! \brief Cast the expr to the given dtype. */
  Expr CastExpr(const Expr expr, const TypeHint& type_hint) {
    auto& scope = scopes_.back();
    CHECK(expr->checked_type_.defined())
        << "The type of " << mnm::ir::AsText(expr) << " is missing";

    auto expr_type = expr->checked_type();
    if (expr_type->IsInstance<TensorTypeNode>()) {
      auto prim_type = TryToPrimType(type_hint);
      auto ttype = Downcast<TensorType>(expr_type);
      if (ttype->dtype == prim_type->dtype) {
        return expr;
      } else {
        auto cast_call = GenCastCall(expr, prim_type->dtype);
        auto new_var = scope.Push(cast_call);
        new_var->checked_type_ = cast_call->checked_type_;
        return new_var;
      }
    } else if (expr_type->IsInstance<TupleTypeNode>()) {
      auto tuple_type = TryToTupleType(type_hint);
      auto cast_call = CastTuple(expr, tuple_type);
      auto new_var = scope.Push(cast_call);
      new_var->checked_type_ = cast_call->checked_type_;
      return new_var;
    } else {
      LOG(FATAL) << "Unsupported type: " << expr_type->GetTypeKey();
    }
  }

  /*! \brief The scope stack of the let list. */
  std::vector<LetList> scopes_;
  /*! \brief The dtype of the generated AMP model. */
  DataType amp_dtype_;
  /*! \brief The output dtype of the generated AMP model. */
  DataType out_dtype_;
  /*! \brief The mapping from let bound var to its expr. */
  std::unordered_map<Var, Expr, ObjectPtrHash, ObjectPtrEqual> let_vars_;
};
}  // namespace auto_cast

TVM_REGISTER_PASS_CONFIG_OPTION("mnm.amp.dtype", String);
TVM_REGISTER_PASS_CONFIG_OPTION("mnm.amp.out_dtype", String);

Pass AutoCast() {
  PassContext pass_ctx = PassContext::Current();
  String amp_dtype = pass_ctx->GetConfig("mnm.amp.dtype", String("float16")).value();
  String out_dtype = pass_ctx->GetConfig("mnm.amp.out_dtype", String("float16")).value();
  DLOG(INFO) << "AMP dtype: " << amp_dtype << ", output dtype: " << out_dtype;
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        auto ret = auto_cast::AutoCastMutator(amp_dtype, out_dtype).Mutate(f);
        return Downcast<Function>(ret);
      };
  auto insert_cast = CreateMNMFunctionPass(pass_func, 0, "AutoCastFunc", {});
  return MNMSequential({InferType(), insert_cast}, "AutoCast");
}

MNM_REGISTER_GLOBAL("mnm.pass_.AutoCast").set_body_typed(AutoCast);
}  // namespace pass
}  // namespace mnm
