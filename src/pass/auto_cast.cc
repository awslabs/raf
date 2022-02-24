/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file auto_cast.cc
 * \brief AutoCast pass
 */
#include <tvm/ir/transform.h>

#include <stack>
#include "raf/op.h"
#include "raf/cache.h"
#include "raf/ir.h"
#include "raf/type.h"
#include "raf/value.h"
#include "raf/pass.h"
#include "./let_list.h"
#include "./common.h"

namespace raf {
namespace pass {
namespace auto_cast {

using namespace raf::ir;
using namespace raf::op;
using namespace raf::value;

using TypeHint = Type;
using TypeHints = Array<TypeHint>;

/*! \brief Cast TypeHint to PrimType. */
inline PrimType AsPrimType(TypeHint type, std::string msg = "") {
  auto prim_type = type.as<PrimTypeNode>();
  msg += (msg == "") ? "" : " ";
  CHECK(prim_type != nullptr) << msg << "Expected PrimType as the type hint, but got "
                              << type->GetTypeKey();
  return GetRef<PrimType>(prim_type);
}

/*! \brief Cast TypeHint to TupleType. */
inline TupleType AsTupleType(TypeHint type, std::string msg = "") {
  auto tuple_type = type.as<TupleTypeNode>();
  msg += (msg == "") ? "" : " ";
  CHECK(tuple_type != nullptr) << msg << "Expected TupleType as the type hint, but got "
                               << type->GetTypeKey();
  return GetRef<TupleType>(tuple_type);
}

/*! \brief Whether the given type hint means don't touch. */
inline bool IsDontTouchTypeHint(TypeHint type) {
  auto prim_type = type.as<PrimTypeNode>();
  return (prim_type && GetRef<PrimType>(prim_type)->dtype.is_void());
}

/*! \brief Generate a type hint that means don't touch. */
inline TypeHint GetDontTouchTypeHint() {
  return PrimType(DataType::Void());
}

/*! \brief Generate a cast call that casts the given expr to the target dtype. */
inline Expr GenCastCall(Expr expr, DataType dtype) {
  static const Op& op = Op::Get("raf.op.cast");
  const auto old_type = Downcast<TensorType>(expr->checked_type());
  std::string target_dtype;
  switch (dtype.code()) {
    case DataType::kFloat:
      target_dtype = "float";
      break;
    case DataType::kUInt:
      target_dtype = "uint";
      break;
    case DataType::kInt:
      target_dtype = "int";
      break;
    default:
      LOG(FATAL) << "Unsupported dtype: " << dtype;
  }
  target_dtype += std::to_string(dtype.bits());
  auto cast_call = Call(op, {expr, MakeConstant(StringValue::make(target_dtype))}, {});
  DataType new_dtype = DataType(String2DLDataType(target_dtype));
  cast_call->checked_type_ = TensorType(old_type->shape, new_dtype);
  return cast_call;
}

/*! \brief Infer the return type of the given call. */
Type InferRetType(const Call& call) {
  static auto fschema = Op::GetAttrMap<op::FRAFSchema>("FRAFSchema");
  auto op_node = call->op.as<OpNode>();
  CHECK(op_node != nullptr) << "Not support closure or global function yet";

  // Generate call values.
  CallValues call_values = CallValues::make();

  // Make argument values.
  Array<Value> arg_values;
  for (const auto& arg : call->args) {
    arg_values.push_back(GetValue(arg));
  }

  call_values->args = fschema[GetRef<Op>(op_node)](arg_values);
  call_values->callee = OpValue::make(GetRef<Op>(op_node));

  static const auto op_type = Op::GetAttrMap<OpType>("OpType");
  auto fty = Downcast<FuncType>(op_type[Downcast<Op>(call->op)]);
  CHECK_EQ(fty->type_constraints.size(), 1);
  TypeInference ti = Downcast<TypeInference>(fty->type_constraints[0]);
  return ti->func(call_values);
}

inline HashKey TypeHintHash(TypeHint type_hint) {
  HashKey key;
  if (auto prim_type = type_hint.as<PrimTypeNode>()) {
    DataType dtype = prim_type->dtype;
    key << DLDataType2String(dtype);
  } else if (auto tuple_type = type_hint.as<TupleTypeNode>()) {
    for (auto field : tuple_type->fields) {
      key << TypeHintHash(field);
    }
  } else {
    LOG(FATAL) << "Unrecorgnized type hint: " << type_hint->GetTypeKey();
    throw;
  }
  return key;
}

/*! \brief Hash of the cast cache. */
struct CastCacheHash {
  std::size_t operator()(const std::pair<Expr, TypeHint>& pair) const {
    HashKey key;
    // Simply use the object hash.
    key << uint64_t(ObjectPtrHash()(pair.first));
    // Cannot use the object hash because type hints are different objects.
    key << TypeHintHash(pair.second);

    std::string str_key(key.byte_vector.begin(), key.byte_vector.end());
    return std::hash<std::string>()(str_key);
  }
};

/*! \brief KeyEqual of the cast cache. */
struct CastCacheEqual {
  bool operator()(const std::pair<Expr, TypeHint>& pair1,
                  const std::pair<Expr, TypeHint>& pair2) const {
    // When two pairs have the same hash key, we simply check if their expr is the same.
    return ObjectPtrEqual()(pair1.first, pair2.first);
  }
};

using CastCache =
    std::unordered_map<std::pair<Expr, TypeHint>, Expr, CastCacheHash, CastCacheEqual>;

class AutoCastMutator : public ExprMutator {
 public:
  AutoCastMutator(const String amp_dtype, const String out_dtype) {
    scopes_.emplace_back(new LetList);
    amp_dtype_ = DataType(String2DLDataType(amp_dtype));
    out_dtype_ = DataType(String2DLDataType(out_dtype));
  }

  /*!
   * \brief Concat missing rule ops and their appearance to a string.
   * \return A string of missing rule ops, or empty if none.
   */
  std::string ListMissRuleOps() {
    if (miss_rule_ops_.empty()) {
      return "";
    }

    std::stringstream ss;
    for (auto pair : miss_rule_ops_) {
      ss << pair.first << " (appear " << pair.second << " times)\n";
    }
    return ss.str();
  }

  Expr VisitExpr_(const LetNode* node) {
    scopes_.emplace_back(new LetList);
    auto scope = scopes_.back().get();
    Expr body;
    do {
      curr_let_ = node->var;
      auto new_value = VisitExpr(node->value);
      auto new_type = new_value->checked_type();

      // If the binding var shares with another var, then its type must align to the shared var.
      // For example, the original IR:
      //  fn (%x: Tensor[(10, 10), float32], %w: Tensor[(10, 10), float32]) {
      //    let %a = matmul(%x, %w); /* Tensor[(10, 10), float32]*/
      //    let %b(share: %x) = %a;
      //    %b;
      //  };
      // becomes:
      //  fn (%x: Tensor[(10, 10), float32], %w: Tensor[(10, 10), float32]) {
      //    let %x_0 = cast(%x, "float16");
      //    let %x_1 = cast(%w, "float16");
      //    let %a = matmul(%x_0, %x_1); /* Tensor[(10, 10), float16]*/
      //    let %x_2 = %a;
      //    let %x_3 = cast(%x_2, "float32");
      //    let %b(share: %x) = %x_3;
      //    %b;
      //  };
      auto extended_var = curr_let_.as<ExtendedVarNode>();
      if (extended_var->may_share.defined()) {
        auto target_type = extended_var->may_share->checked_type().as<TensorTypeNode>();
        auto curr_type = new_type.as<TensorTypeNode>();
        CHECK(target_type != nullptr && curr_type != nullptr);
        if (target_type->dtype != curr_type->dtype) {
          auto uncast_var = scope->Push(new_value);
          uncast_var->checked_type_ = new_type;
          new_type = GetRef<TensorType>(target_type);
          new_value = scope->Push(GenCastCall(uncast_var, target_type->dtype));
          new_value->checked_type_ = new_type;
        }
      }

      curr_let_->checked_type_ = new_type;
      scope->Push(curr_let_, new_value);
      let_vars_to_orig_type_.emplace(curr_let_, node->value->checked_type());
      let_vars_.emplace(curr_let_, new_value);

      body = node->body;
      node = body.as<LetNode>();
    } while (node);
    auto new_body = VisitExpr(body);

    // Cast output tensors if needed.
    TypeHint ret_type_hint = GenOutputTypeHint(new_body->checked_type());
    new_body = CastExpr(new_body, ret_type_hint);
    auto ret = scopes_.back()->Get(new_body);
    ret->checked_type_ = new_body->checked_type();
    scopes_.pop_back();
    return ret;
  }

  Expr VisitExpr_(const CallNode* node) {
    static const Op& cast_op = Op::Get("raf.op.cast");

    // If the argument is a cast, then we use the uncasted one to generate the type hints,
    // so that the infer cast ops can follow the uncasted argument and minimize the cast op.
    Array<Expr> uncasted_call_args;
    for (size_t i = 0; i < node->args.size(); ++i) {
      auto arg_var = node->args[i].as<VarNode>();
      if (arg_var != nullptr && let_vars_.count(GetRef<Var>(arg_var)) > 0) {
        auto arg_call = let_vars_[GetRef<Var>(arg_var)].as<CallNode>();
        if (arg_call && arg_call->op->IsInstance<OpNode>()) {
          auto arg_op = arg_call->op.as<OpNode>();
          if (GetRef<Op>(arg_op) == cast_op) {
            uncasted_call_args.push_back(arg_call->args[0]);
            continue;
          }
        }
      }
      uncasted_call_args.push_back(node->args[i]);
    }
    TypeHints type_hints =
        GenTypeHints(uncasted_call_args, node->checked_type(), amp_dtype_, node->op);

    auto op = Downcast<Op>(node->op);
    CHECK_EQ(type_hints.size(), node->args.size())
        << "Type hint number and argument size of " << op->name << " are mismatching";

    // Special process for existing cast ops.
    if (op == cast_op) {
      auto in_type = node->args[0]->checked_type().as<TensorTypeNode>();
      auto ret_type = node->checked_type().as<TensorTypeNode>();
      CHECK(node->args[0]->IsInstance<VarNode>() || node->args[0]->IsInstance<ConstantNode>());
      CHECK(in_type != nullptr && ret_type != nullptr);

      // This case op is not required anymore because its argument already produces the AMP dtype.
      if (in_type->dtype == ret_type->dtype) {
        return node->args[0];
      }

      // Add the cast op to the reversed cache to avoid generating back-to-back cast ops.
      auto pair_key = std::make_pair(curr_let_, PrimType(in_type->dtype));
      reversed_cast_cache_[pair_key] = node->args[0];
      auto new_call = Call(op, node->args, node->attrs, node->type_args);
      new_call->checked_type_ = node->checked_type();
      return new_call;
    }

    // Try to lower to tvm dialect op and get op pattern
    int pattern = kOpaque;
    auto tvm_op = OpDialect::Lower(op, "tvm");
    if (tvm_op.defined()) {
      pattern = GetOpAttr<TOpPattern>(tvm_op, "TOpPattern");
    }

    // If the fusion pattern of this op is elementwise or broadcast, we disable the cache
    // to make sure they their argument cast op will not be reused and can fused together.
    // TODO(comaniac): Let the recompute or fusion pass work on this.
    bool use_cache = pattern > kInjective;

    Array<Expr> call_args;
    for (size_t i = 0; i < node->args.size(); ++i) {
      UpdateCurrExprStr(op->name, i);
      call_args.push_back(CastExpr(VisitExpr(node->args[i]), type_hints[i], use_cache));
    }

    auto new_call = Call(op, call_args, node->attrs, node->type_args);
    UpdateCurrExprStr(op->name, -1);
    new_call->checked_type_ = InferRetType(new_call);

    UpdateCurrExprStr("", 0);
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
  /*! \brief A helper function to generate a string of current processing node and argument
   * to form a better error message. arg_idx=-1 means the return type.
   */
  inline void UpdateCurrExprStr(std::string name, int arg_idx) {
    curr_call_n_arg_str_ = "";
    if (name == "") {
      return;
    }

    curr_call_n_arg_str_ = name;
    if (arg_idx == -1) {
      curr_call_n_arg_str_ += ".ret";
    } else {
      curr_call_n_arg_str_ += ".arg" + std::to_string(arg_idx);
    }
  }

  /*! \brief Generate the type hints for the output expression.
   * We cast the output tensors with AMP dtype to the user-specified output dtype.
   * Note that we do not touch output tensors with other dtypes, because they may not be the
   * data output (e.g., mean and variant of batch_norm), but this approach cannot catch the
   * case that the last op is a never-cast op (e.g., erf). In this case, the output tensor
   * remains float32 even user specified float16.
   */
  TypeHint GenOutputTypeHint(const Type type) {
    if (auto tuple_type = type.as<TupleTypeNode>()) {
      TypeHints field_types;
      for (auto field_type : tuple_type->fields) {
        field_types.push_back(GenOutputTypeHint(field_type));
      }
      return TupleType(field_types);
    }

    auto ttype = type.as<TensorTypeNode>();
    CHECK(ttype) << "Expected tensor type node, but got " << type->GetTypeKey()
                 << " when processsing return type";
    if (ttype->dtype != amp_dtype_) {
      // If it is not the AMP dtype, it means this tensor should not be touched by AutoCast.
      return GetDontTouchTypeHint();
    }
    return PrimType(out_dtype_);
  }

  /*! \brief Ideally, the default type hint should cast input tensors in float while
   * leaving attributes untouched. However, we cannot differentiate inputs and attributes
   * as both of them are arguments. Thus, default type hint is only used to make sure
   * this pass can run through the whole graph without crashing, so that we can throw out
   * as error with all missing ops.
   */
  TypeHints GetDefaultTypeHint(const Array<Expr>& args, const Type& ret_type,
                               const DataType target_dtype) {
    TypeHints type_hints;
    for (size_t i = 0; i < args.size(); ++i) {
      type_hints.push_back(GetDontTouchTypeHint());
    }
    return type_hints;
  }

  /*! \brief Generate the type hints of an op by looking at its type hint rules. */
  TypeHints GenTypeHints(const Array<Expr>& args, const Type& ret_type, const DataType target_dtype,
                         const Expr op_node) {
    static auto frule = Op::GetAttrMap<op::FRAFCastRule>("FRAFCastRule");
    CHECK(op_node.as<OpNode>() != nullptr)
        << "AutoCast does not support closure yet: " << raf::ir::AsText(op_node);
    const Op op = Downcast<Op>(op_node);

    if (frule.count(op)) {
      return frule[op](args, ret_type, DLDataType2String(target_dtype));
    } else {
      miss_rule_ops_[op]++;
    }
    return GetDefaultTypeHint(args, ret_type, target_dtype);
  }

  /*! \brief Get the expression of the given let binding var. If the given expr is not a VarNode,
   * then simply return itself.
   */
  inline Expr GetBindExpr(const Expr expr) {
    if (expr->IsInstance<VarNode>()) {
      return let_vars_[Downcast<Var>(expr)];
    }
    return expr;
  }

  /*! \brief Generate a tuple of casted tensors. */
  Expr CastTuple(const Expr arg, const TypeHint type_hint, bool use_cache = true) {
    auto scope = scopes_.back().get();

    auto expr = GetBindExpr(arg);
    if (expr->IsInstance<ConstantNode>()) {
      CHECK(IsDontTouchTypeHint(type_hint))
          << curr_call_n_arg_str_ << " has an illegal type hint: " << type_hint->GetTypeKey()
          << ". Cannot cast constant tuple";
      return expr;
    }

    auto tuple_type = Downcast<TupleType>(expr->checked_type());
    auto tuple_type_hint = AsTupleType(type_hint, curr_call_n_arg_str_);
    CHECK_EQ(tuple_type->fields.size(), tuple_type_hint->fields.size())
        << curr_call_n_arg_str_ << " has an illegal tuple type hint: Expected "
        << tuple_type->fields.size() << " fields for the following expression, but got "
        << tuple_type_hint->fields.size() << "\n"
        << raf::ir::AsText(expr);

    // Find the root tuple in case the expr has a tuple type but in a nested tuple.
    while (auto tgi = expr.as<TupleGetItemNode>()) {
      expr = GetBindExpr(tgi->tuple);
      expr = Downcast<Tuple>(expr)->fields[tgi->index];
      expr = GetBindExpr(expr);
    }

    // The tuple is generated by an op, such as batch_norm and split.
    if (!expr->IsInstance<TupleNode>()) {
      // Check whether any of its element needs to be casted.
      bool need_new_tuple = false;
      for (size_t i = 0; i < tuple_type->fields.size(); ++i) {
        auto ttype = tuple_type->fields[i].as<TensorTypeNode>();
        CHECK(ttype != nullptr) << "Does not support an op that outputs a nested tuple";
        auto target_dtype = AsPrimType(tuple_type_hint->fields[i], curr_call_n_arg_str_)->dtype;
        if (!IsDontTouchTypeHint(tuple_type_hint->fields[i]) && (ttype->dtype != target_dtype)) {
          need_new_tuple = true;
          LOG(INFO) << "need new tuple " << ttype->dtype << " vs " << target_dtype;
          break;
        }
      }

      // Generate a new tuple. For example:
      // Before:
      // let %1 = split(%0, 2, axis=0);
      //
      // After:
      // let %1 = split(%0, 2, axis=0);
      // let %2 = %1.0;
      // let %3 = %1.1;
      // let %4 = (%2, %3);
      if (need_new_tuple) {
        Array<Expr> fields;
        Array<Type> field_types;
        for (size_t i = 0; i < tuple_type->fields.size(); ++i) {
          auto ttype = Downcast<TensorType>(tuple_type->fields[i]);
          auto tgi = scope->Push(TupleGetItem(expr, i));
          tgi->checked_type_ = ttype;
          fields.push_back(tgi);
          field_types.push_back(tgi->checked_type_);
        }
        expr = Tuple(fields);
        expr->checked_type_ = TupleType(field_types);
        auto tuple_var = scope->Push(expr);
        tuple_var->checked_type_ = expr->checked_type_;
      } else {
        // Otherwise do nothing.
        return expr;
      }
    }

    auto tuple = expr.as<TupleNode>();
    CHECK(tuple != nullptr) << "Internal error";
    Array<Expr> fields;
    Array<Type> field_types;
    for (size_t i = 0; i < tuple->fields.size(); ++i) {
      auto field = tuple->fields[i];
      auto field_type = field->checked_type();
      Expr new_expr;
      if (field_type->IsInstance<TensorTypeNode>()) {
        new_expr = CastExpr(field, tuple_type_hint->fields[i], use_cache);
      } else if (field_type->IsInstance<TupleTypeNode>()) {
        new_expr = CastTuple(field, tuple_type_hint->fields[i], use_cache);
      } else {
        LOG(FATAL) << "Unsupported field type: " << field_type->GetTypeKey();
      }
      auto new_var = scope->Push(new_expr);
      new_var->checked_type_ = new_expr->checked_type_;
      fields.push_back(new_var);
      field_types.push_back(new_var->checked_type_);
    }
    auto new_tuple = Tuple(fields);
    new_tuple->checked_type_ = TupleType(field_types);
    return new_tuple;
  }

  /*! \brief Cast the expr based on the given type hint. */
  Expr CastExpr(const Expr expr, const TypeHint& type_hint, bool use_cache = true) {
    auto scope = scopes_.back().get();
    CHECK(expr->checked_type_.defined()) << "Missing type:\n" << raf::ir::AsText(expr);

    auto expr_type = expr->checked_type();
    if (expr_type->IsInstance<TupleTypeNode>()) {
      auto cast_call = CastTuple(expr, type_hint, use_cache);
      if (expr != cast_call) {
        auto new_var = scope->Push(cast_call);
        new_var->checked_type_ = cast_call->checked_type_;
        return new_var;
      }
      return expr;
    }

    CHECK(expr_type->IsInstance<TensorTypeNode>())
        << "Expected tensor type node, but got " << expr_type->GetTypeKey();
    auto ttype = Downcast<TensorType>(expr_type);
    auto target_dtype = AsPrimType(type_hint, curr_call_n_arg_str_)->dtype;
    if (IsDontTouchTypeHint(type_hint)) {
      // Set the target dtype to be the original dtype.
      Type orig_type;
      auto var = expr.as<VarNode>();
      if (var && let_vars_to_orig_type_.count(GetRef<Var>(var)) > 0) {
        // Find the original type of this let binding var.
        orig_type = let_vars_to_orig_type_[GetRef<Var>(var)];
      } else {
        // If expr is a constant or an input tensor, then it is in its original type.
        orig_type = expr->checked_type();
      }
      CHECK(orig_type->IsInstance<TensorTypeNode>())
          << "Expected tensor type but got " << orig_type->GetTypeKey();
      target_dtype = Downcast<TensorType>(orig_type)->dtype;
    }

    auto pair_key = std::make_pair(expr, type_hint);
    if (reversed_cast_cache_.count(pair_key) > 0) {
      // Directly use the uncasted expression if available when the expr is a cast call
      // but we want an uncasted expression.
      return reversed_cast_cache_[pair_key];
    }
    if (ttype->dtype == target_dtype) {
      return expr;
    }
    if (use_cache && cast_cache_.count(pair_key) > 0) {
      // Reuse the cast if this expression has been casted already.
      return cast_cache_[pair_key];
    }

    auto cast_call = GenCastCall(expr, target_dtype);
    auto new_var = scope->Push(cast_call);
    new_var->checked_type_ = cast_call->checked_type_;
    if (use_cache) {
      // If not using cache, then we make sure this new cast op is only used by one op.
      cast_cache_[pair_key] = new_var;
    }
    return new_var;
  }

  /*! \brief The scope stack of the let list. */
  std::vector<std::unique_ptr<LetList>> scopes_;
  /*! \brief The dtype of the generated AMP model. */
  DataType amp_dtype_;
  /*! \brief The output dtype of the generated AMP model. */
  DataType out_dtype_;
  /*! \brief The current processing call node op name and argument index (for debugging purpose). */
  std::string curr_call_n_arg_str_ = "";
  /*! \brief Map from expr and type hint to the casted var. */
  CastCache cast_cache_;
  /*! \brief Map from the casted var and type hint to the uncasted var. */
  CastCache reversed_cast_cache_;
  /*! \brief The current processing let-binding var. */
  Var curr_let_;
  /*! \brief The mapping from let bound var to its expr. */
  std::unordered_map<Var, Expr, ObjectPtrHash, ObjectPtrEqual> let_vars_;
  /*! \brief The mapping from let bound var to its original type. */
  std::unordered_map<Var, Type, ObjectPtrHash, ObjectPtrEqual> let_vars_to_orig_type_;
  /*! \brief Map from ops that the miss casting rule to appearance. */
  std::unordered_map<Op, int, ObjectPtrHash, ObjectPtrEqual> miss_rule_ops_;
};
}  // namespace auto_cast

TVM_REGISTER_PASS_CONFIG_OPTION("raf.amp.dtype", String);
TVM_REGISTER_PASS_CONFIG_OPTION("raf.amp.out_dtype", String);

Pass AutoCast() {
  PassContext pass_ctx = PassContext::Current();
  String amp_dtype = pass_ctx->GetConfig("raf.amp.dtype", String("float16")).value();
  String out_dtype = pass_ctx->GetConfig("raf.amp.out_dtype", String("float16")).value();
  DLOG(INFO) << "AMP dtype: " << amp_dtype << ", output dtype: " << out_dtype;
  TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func = [=](Function f, IRModule m,
                                                                             PassContext pc) {
    auto mutator = auto_cast::AutoCastMutator(amp_dtype, out_dtype);
    auto ret = mutator.Mutate(f);
    std::string miss_rule_op_str = mutator.ListMissRuleOps();
    if (!miss_rule_op_str.empty()) {
      LOG(FATAL) << "One or more ops missed the casting rule:\n" << miss_rule_op_str;
      throw;
    }
    return Downcast<Function>(ret);
  };
  auto insert_cast = CreateRAFFunctionPass(pass_func, 0, "AutoCastFunc", {});
  return RAFSequential({InferType(), insert_cast, InferType(), DeadCodeElimination()}, "AutoCast");
}

RAF_REGISTER_GLOBAL("raf.pass_.AutoCast").set_body_typed(AutoCast);
}  // namespace pass
}  // namespace raf
