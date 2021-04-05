/*!
 * Copyright (c) 2020 by Contributors
 * \file type_infer.cc
 * \brief Type inference pass
 */

#include <tvm/ir/module.h>
#include <tvm/ir/type_functor.h>
#include <tvm/tir/op.h>
#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/pass.h"
#include "mnm/pass_manager.h"
#include "mnm/binding.h"
#include "mnm/type.h"
#include "../op/ty/utils.h"
#include "tvm/node/structural_equal.h"

namespace mnm {
namespace pass {
namespace type_infer {

using namespace mnm::ir;
using namespace mnm::op;
using namespace mnm::value;
using namespace mnm::type;
using namespace tvm;
using namespace tvm::relay;
using namespace tvm::transform;
using tvm::TypeFunctor;

Type Unify(const Type& src, const Type& dst);

Value GetValue(Type type);

Value GetValue(Expr expr);

#define MNM_NODE_NOT_IMPL(NodeType)                     \
  Expr VisitExpr_(const NodeType* node) override {      \
    LOG(FATAL) << "NotImplementedError: " << #NodeType; \
    throw;                                              \
  }

class TypeInferencer : public ExprMutator {
 public:
  MNM_NODE_NOT_IMPL(RefReadNode)
  MNM_NODE_NOT_IMPL(RefWriteNode)
  MNM_NODE_NOT_IMPL(RefCreateNode)

 public:
  TypeInferencer(IRModule& mod) : mod_(mod) {
  }

  Type GetValueType(const Value& v) {
    return op::type::GetType(v);
  }

  Expr VisitExpr_(const VarNode* op) override {
    if (op->type_annotation.defined()) {
      op->checked_type_ = op->type_annotation;
    } else if (!op->checked_type_.defined()) {
      op->checked_type_ = IncompleteType(kType);
    }
    return GetRef<Var>(op);
  }

  Expr VisitExpr_(const GlobalVarNode* op) override {
    CHECK(mod_.defined());
    return std::move(GetRef<GlobalVar>(op));
  }

  CallValues SchemaToValue(Array<Expr> args, const OpNode* op) {
    static auto fschema = Op::GetAttrMap<op::FMNMSchema>("FMNMSchema");
    CallValues call_values = CallValues::make();
    Array<Value> arg_values;
    for (const auto& arg : args) {
      if (var_value_map_.count(arg.as<VarNode>())) {
        arg_values.push_back(GetValue(var_value_map_[arg.as<VarNode>()]));
      } else {
        arg_values.push_back(GetValue(arg));
      }
    }
    call_values->args = fschema[GetRef<Op>(op)](arg_values);
    call_values->callee = OpValue::make(GetRef<Op>(op));
    return call_values;
  }

  Expr VisitExpr_(const CallNode* call) override {
    Array<Expr> args;
    for (const auto& arg : call->args) {
      args.push_back(VisitExpr(arg));
    }
    const OpNode* opn = call->op.as<OpNode>();
    static const auto declare_op = Op::GetAttrMap<op::FMNMDeclare>("FMNMDeclare");
    // We do constant-folding for shape-related operators by invoking their declare function,
    // because they produce shape information which is required by type inference.
    // The arguments (SchemaToValue(args)) passed to declare function
    // can be either types or tensor values, depends on whether
    // they have already been evaluated/constant-folded.
    // Therefore it is essential to deal with both cases in their declare functions.
    static std::unordered_set<std::string> shape_list{
        "mnm.op.shape", "mnm.op.get_reduce_axis", "mnm.op.get_kept_dims", "mnm.op.concatenate_dx"};
    if (opn && shape_list.count(opn->name)) {
      CallValues call_values = SchemaToValue(args, opn);
      declare_op[GetRef<Op>(opn)](call_values);
      if (call_values->out.defined()) {
        Expr re = ir::MakeConstant(call_values->out);
        return VisitExpr(re);
      }
    }
    if (const FunctionNode* fn = call->op.as<FunctionNode>()) {
      CHECK(fn->HasNonzeroAttr(tvm::relay::attr::kPrimitive))
          << "A primitive function is expected in " << call->op;
      CHECK_EQ(call->args.size(), fn->params.size());
      for (size_t n = call->args.size(), i = 0; i < n; ++i) {
        Expr arg = VisitExpr(call->args[i]);
        const auto* v = arg.as<VarNode>();
        if (v && var_value_map_.count(v)) {
          var_value_map_[fn->params[i].get()] = var_value_map_[v];
        } else {
          var_value_map_[fn->params[i].get()] = arg;
        }
      }
    }
    Expr op = VisitExpr(call->op);
    Call ret = Call(op, args, call->attrs, call->type_args);
    if (const FunctionNode* fn = ret->op.as<FunctionNode>()) {
      ret->checked_type_ = InferClosure(ret, fn);
    } else if (const GlobalVarNode* gvn = ret->op.as<GlobalVarNode>()) {
      ret->checked_type_ =
          InferClosure(ret, Downcast<Function>(mod_->Lookup(GetRef<GlobalVar>(gvn))).get());
    } else if (const OpNode* opn = ret->op.as<OpNode>()) {
      ret->checked_type_ = InferPrimitive(ret, opn);
    } else if (const VarNode* var_node = ret->op.as<VarNode>()) {
      // The var node can be a result of the output type of a func call. A var node
      // here is valid if it points to a function. Check that the type is a FuncType
      // and the args of the Call match the type of the FuncType. If yes, return the
      // FuncType's ret_type.
      const FuncTypeNode* fty_node = ret->op->checked_type_.as<FuncTypeNode>();
      CHECK(fty_node);
      for (size_t i = 0; i < fty_node->arg_types.size(); i++) {
        Type arg_type = fty_node->arg_types[i];
        CHECK(tvm::StructuralEqual()(arg_type, ret->args[0]->checked_type()));
      }
      ret->checked_type_ = fty_node->ret_type;
    } else {
      LOG(FATAL) << "Invalid op type: " << call->op->GetTypeKey();
    }
    return ret;
  }

  Type InferPrimitive(const Call& call, const OpNode* op) {
    // Only type inference from leaf to root is supported.
    // Thus incomplete inputs will not be inferred from outputs.
    // Instead, the incompleteness propogates.
    for (const auto& arg : call->args) {
      const Type& type = arg->checked_type();
      if (const auto* itn = type.as<IncompleteTypeNode>()) {
        return IncompleteType(kType);
      }
    }
    CallValues call_values = SchemaToValue(call->args, op);
    // invoke type inference
    auto fty = Downcast<FuncType>(op->checked_type());
    CHECK_EQ(fty->type_constraints.size(), 1);
    TypeInference ti = Downcast<TypeInference>(fty->type_constraints[0]);
    return ti->func(call_values);
  }

  Type InferClosure(const Call& call, const FunctionNode* fn) {
    // TODO(@hzfan): perform template param deduction to eliminate type_params
    FuncType fty = Downcast<FuncType>(fn->checked_type());
    CHECK_EQ(call->args.size(), fty->arg_types.size());
    for (size_t i = 0; i < call->args.size(); ++i) {
      CHECK(StructuralEqual()(call->args[i]->checked_type(), fty->arg_types[i]))
          << "Type of argument and function parameter mismatch: " << call->args[i]->checked_type()
          << " vs " << fty->arg_types[i];
    }
    return fty->ret_type;
  }

  Expr VisitExpr_(const RelayConstantNode* op) override {
    const ConstantNode* node = static_cast<const ConstantNode*>(op);
    auto const_data = node->value;
    // check if the constant is null
    if (const_data.defined()) {
      if (const ArrayNode* arr = const_data.as<ArrayNode>()) {
        Array<PrimExpr> shape;
        for (const auto& it : *arr) {
          CHECK(it->IsInstance<IntImmNode>());
          shape.push_back(static_cast<int32_t>((Downcast<IntImm>(it))->value));
        }
        op->checked_type_ = TensorType(shape, op->tensor_type()->dtype);
      } else {
        op->checked_type_ = GetValueType(Downcast<Value>(const_data));
      }
    } else {
      // fake type info
      op->checked_type_ = TensorType::Scalar(DataType::Int(64));
    }
    return GetRef<Expr>(op);
  }

  Expr VisitExpr_(const IfNode* node) override {
    Expr cond = VisitExpr(node->cond);
    Expr true_branch = VisitExpr(node->true_branch);
    Expr false_branch = VisitExpr(node->false_branch);
    Expr ret = If(cond, true_branch, false_branch);
    ret->checked_type_ = Unify(true_branch->checked_type(), false_branch->checked_type());
    return ret;
  }

  Expr VisitExpr_(const LetNode* op) override {
    Expr ovalue = op->value;
    Var var = op->var;
    Expr value = VisitExpr(ovalue);
    if (value.as<ConstantNode>()) {
      memo_[var] = value;
      return VisitExpr(op->body);
    }
    const VarNode* v = value.as<VarNode>();
    if (v && var_value_map_.count(v)) {
      var_value_map_[op->var.get()] = var_value_map_[v];
    } else {
      var_value_map_[op->var.get()] = value;
    }
    var->checked_type_ = value->checked_type();
    Expr body = VisitExpr(op->body);
    Let let(var, value, body);
    let->checked_type_ = body->checked_type();
    return let;
  }

  Expr VisitExpr_(const TupleNode* op) override {
    Array<Expr> fields;
    Array<Type> types;
    for (const auto& e : op->fields) {
      auto f = VisitExpr(e);
      fields.push_back(f);
      types.push_back(f->checked_type());
    }
    Tuple ret(fields);
    ret->checked_type_ = TupleType(types);
    return ret;
  }

  Expr VisitExpr_(const TupleGetItemNode* op) override {
    auto tup = VisitExpr(op->tuple);
    TupleGetItem ret(tup, op->index);
    ret->checked_type_ = Downcast<TupleType>(tup->checked_type())->fields[op->index];
    return ret;
  }

  Expr VisitExpr_(const OpNode* op) override {
    static const auto op_type = Op::GetAttrMap<OpType>("OpType");
    op->checked_type_ = op_type[GetRef<Op>(op)];
    return GetRef<Expr>(op);
  }

  Expr VisitExpr_(const FunctionNode* op) override {
    Array<Var> params;
    Array<Type> param_types;
    for (const auto& p : op->params) {
      Var param = Downcast<Var>(VisitExpr(p));
      params.push_back(param);
      param_types.push_back(param->checked_type());
    }
    Expr body = VisitExpr(op->body);
    Type ret_type =
        op->ret_type.defined() ? Unify(body->checked_type(), op->ret_type) : body->checked_type();
    Function func(params, body, ret_type, op->type_params, op->attrs);
    func->checked_type_ = FuncType(param_types, ret_type, op->type_params, {});
    return func;
  }

 private:
  IRModule mod_;
  // The var_value_map_ is used to track Let binding Expr
  // E.g. Let %a = %b; Let %c = some_op(%a)
  // The var_value_map_ will feed %b to some_op
  std::unordered_map<const VarNode*, Expr> var_value_map_;
};

class Unifier : public TypeFunctor<Type(const Type&, const Type&)> {
 public:
  Type Unify(const Type& src, const Type& dst) {
    if (src.as<IncompleteTypeNode>() || !src.defined()) {
      return dst;
    } else if (dst.as<IncompleteTypeNode>() || !dst.defined()) {
      return src;
    } else {
      Type resolved = this->VisitType(src, dst);
      CHECK(resolved.defined()) << "unable to unify: "
                                << "`" << PrettyPrint(src) << "` and `" << PrettyPrint(dst) << "`";
      return resolved;
    }
  }

  // default: unify only if structural-equal
  Type VisitTypeDefault_(const Object* op, const Type& tn) final {
    ObjectRef nr = GetRef<ObjectRef>(op);
    Type t1 = GetRef<Type>(nr.as<tvm::relay::TypeNode>());
    if (!tvm::StructuralEqual()(t1, tn)) {
      return Type(nullptr);
    }
    return t1;
  }

  IndexExpr UnifyDim(const IndexExpr& lhs, const IndexExpr& rhs) {
    if (lhs.same_as(rhs)) {
      return lhs;
    }
    if (lhs.as<AnyNode>() || rhs.as<AnyNode>()) {
      return Any();
    }

    auto left_index0 = lhs.as<tvm::tir::VarNode>();
    auto right_index0 = rhs.as<tvm::IntImmNode>();
    if (left_index0 && right_index0) {
      return rhs;
    }

    auto left_index1 = lhs.as<tvm::IntImmNode>();
    auto right_index1 = rhs.as<tvm::tir::VarNode>();
    if (left_index1 && right_index1) {
      return lhs;
    }

    auto left_index2 = lhs.as<tvm::IntImmNode>();
    auto right_index2 = rhs.as<tvm::IntImmNode>();
    if (left_index2 && right_index2 && left_index2->value == right_index2->value) {
      return lhs;
    }

    return tvm::PrimExpr();
  }

  Type VisitType_(const TensorTypeNode* op, const Type& tn) final {
    const auto* tt_node = tn.as<TensorTypeNode>();
    CHECK(tt_node);
    auto tt1 = GetRef<TensorType>(op);
    auto tt2 = GetRef<TensorType>(tt_node);
    if (tvm::StructuralEqual()(tt1, tt2)) {
      return std::move(tt1);
    }
    CHECK(tt1->dtype == tt2->dtype);

    tvm::Array<IndexExpr> shape;
    CHECK_EQ(tt1->shape.size(), tt2->shape.size())
        << "tensor type `" << PrettyPrint(tt1) << "` has " << tt1->shape.size()
        << " dimensions, while `" << PrettyPrint(tt2) << "` has " << tt2->shape.size()
        << " dimensions";

    CHECK_EQ(tt1->shape.size(), tt2->shape.size());
    for (size_t i = 0; i < tt1->shape.size(); i++) {
      auto dim = UnifyDim(tt1->shape[i], tt2->shape[i]);
      CHECK(dim.defined());
      shape.push_back(dim);
    }
    return TensorType(shape, tt1->dtype);
  }

  Type VisitType_(const TupleTypeNode* op, const Type& tn) final {
    const auto* ttn = tn.as<TupleTypeNode>();
    CHECK(ttn && op->fields.size() == ttn->fields.size());

    TupleType tt1 = GetRef<TupleType>(op);
    TupleType tt2 = GetRef<TupleType>(ttn);

    std::vector<Type> new_fields;
    for (size_t i = 0; i < tt1->fields.size(); i++) {
      Type field = Unify(tt1->fields[i], tt2->fields[i]);
      new_fields.push_back(field);
    }
    return TupleType(new_fields);
  }

  Type VisitType_(const FuncTypeNode* op, const Type& tn) final {
    const auto* ftn = tn.as<FuncTypeNode>();
    CHECK(ftn && op->arg_types.size() == ftn->arg_types.size() &&
          op->type_constraints.size() == ftn->type_constraints.size());

    // without loss of generality, suppose op->type_params.size() >= ftn->type_params.size().
    if (op->type_params.size() < ftn->type_params.size()) {
      return VisitType_(ftn, GetRef<FuncType>(op));
    }

    // remap type vars so they match
    Map<TypeVar, Type> subst_map;
    tvm::Array<TypeVar> ft_type_params;
    for (size_t i = 0; i < ftn->type_params.size(); ++i) {
      subst_map.Set(op->type_params[i], ftn->type_params[i]);
      ft_type_params.push_back(op->type_params[i]);
    }

    for (size_t i = ftn->type_params.size(); i < op->type_params.size(); ++i) {
      subst_map.Set(op->type_params[i], IncompleteType(kType));
    }

    FuncType ft = FuncType(op->arg_types, op->ret_type, ft_type_params, op->type_constraints);
    auto ft1 = Downcast<FuncType>(Bind(ft, subst_map));
    auto ft2 = GetRef<FuncType>(ftn);

    Type ret_type = Unify(ft1->ret_type, ft2->ret_type);

    std::vector<Type> arg_types;
    for (size_t i = 0; i < ft2->arg_types.size(); ++i) {
      Type arg_type = Unify(ft1->arg_types[i], ft2->arg_types[i]);
      arg_types.push_back(arg_type);
    }

    std::vector<TypeConstraint> type_constraints;
    for (size_t i = 0; i < ft1->type_constraints.size(); ++i) {
      Type unified_constraint = Unify(ft1->type_constraints[i], ft2->type_constraints[i]);
      const auto* tcn = unified_constraint.as<TypeConstraintNode>();
      CHECK(tcn) << "Two type constraints unified into a non-constraint?"
                 << ft1->type_constraints[i] << " and " << ft2->type_constraints[i];
      type_constraints.push_back(GetRef<TypeConstraint>(tcn));
    }

    return FuncType(arg_types, ret_type, ft2->type_params, type_constraints);
  }

  Type VisitType_(const TypeCallNode* op, const Type& tn) override {
    const auto* tcn = tn.as<TypeCallNode>();
    if (!tcn || tcn->args.size() != op->args.size()) {
      return Type();
    }

    Type func = Unify(op->func, tcn->func);
    tvm::Array<Type> args;
    for (size_t i = 0; i < op->args.size(); i++) {
      args.push_back(Unify(op->args[i], tcn->args[i]));
    }
    return TypeCall(func, args);
  }
};

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

  Value VisitExprDefault_(const Object* op) {
    const auto* e = static_cast<const ExprNode*>(op);
    return GetValue(e->checked_type());
  }
};

Value GetValue(Type type) {
  return TypeGetter()(type);
}

Value GetValue(Expr expr) {
  return ValueGetter()(expr);
}

Type Unify(const Type& src, const Type& dst) {
  Unifier unifier;
  return unifier.Unify(src, dst);
}

}  // namespace type_infer

void AddGlobalTypes(ir::IRModule mod) {
  std::vector<std::pair<ir::GlobalVar, ir::Function> > updates;
  for (const auto& it : mod->functions) {
    if (auto* func_node = it.second.as<ir::FunctionNode>()) {
      ir::Function func = ir::Function(ir::make_object<ir::FunctionNode>(*func_node));
      func->checked_type_ = func->func_type_annotation();
      updates.push_back({it.first, tvm::runtime::Downcast<ir::Function>(func)});
    }
  }

  for (const auto& pair : updates) {
    mod->Add(pair.first, pair.second, true);
  }
}

ir::Expr InferType(ir::Expr func) {
  auto mod = ir::GlobalModule();
  return type_infer::TypeInferencer(mod).VisitExpr(func);
}

Pass InferType() {
  auto pass_info = PassInfo(0, "InferType", {});
  return CreateModulePass(
      [=](IRModule mod, const PassContext& pass_ctx) {
        DLOG(INFO) << "pass::InferType";
        ir::IRModule updated_mod = ir::IRModule(mod->functions);
        AddGlobalTypes(updated_mod);
        auto ti = type_infer::TypeInferencer(updated_mod);
        for (auto kv : updated_mod->functions) {
          if (kv.second.as<ir::FunctionNode>()) {
            auto func = tvm::runtime::Downcast<ir::Function>(ti.VisitExpr(kv.second));
            updated_mod->Add(kv.first, func, true);
          }
        }
        return updated_mod;
      },
      0, "InferType", {});
}

MNM_REGISTER_GLOBAL("mnm.pass_.InferType").set_body_typed([]() { return InferType(); });

}  // namespace pass
}  // namespace mnm
