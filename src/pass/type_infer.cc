/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file type_infer.cc
 * \brief Type inference pass
 */

#include <tvm/ir/module.h>
#include <tvm/ir/type_functor.h>
#include <tvm/tir/op.h>
#include "raf/op.h"
#include "raf/ir_ext.h"
#include "raf/pass.h"
#include "raf/pass_manager.h"
#include "raf/binding.h"
#include "raf/type.h"
#include "./common.h"
#include "../op/ty/utils.h"
#include "tvm/node/structural_equal.h"

namespace raf {
namespace pass {
namespace type_infer {

using namespace raf::op;
using namespace raf::value;

Type Unify(const Type& src, const Type& dst);

#define RAF_NODE_NOT_IMPL(NodeType)                     \
  Expr VisitExpr_(const NodeType* node) override {      \
    LOG(FATAL) << "NotImplementedError: " << #NodeType; \
    throw;                                              \
  }

class TypeInferencer : public ExprMutator {
 public:
  RAF_NODE_NOT_IMPL(RefReadNode)
  RAF_NODE_NOT_IMPL(RefWriteNode)
  RAF_NODE_NOT_IMPL(RefCreateNode)

 public:
  TypeInferencer(IRModule& mod) : mod_(mod) {
  }

  Type GetValueType(const Value& v) {
    return op::GetType(v);
  }

  Expr VisitExpr_(const VarNode* op) override {
    auto var = GetRef<Var>(op);
    if (closure_param_map_.count(var) > 0) {
      // Use the updated closure parame var.
      var = closure_param_map_[var];
    }
    if (!var->checked_type_.defined()) {
      if (var->type_annotation.defined()) {
        var->checked_type_ = var->type_annotation;
      } else {
        var->checked_type_ = IncompleteType(kType);
      }
    }
    return var;
  }

  Expr VisitExpr_(const GlobalVarNode* op) override {
    CHECK(mod_.defined());
    CHECK(mod_->ContainGlobalVar(op->name_hint))
        << "Module does not contain " << GetRef<GlobalVar>(op);
    Expr func = mod_->Lookup(GetRef<GlobalVar>(op));
    func = VisitExpr(func);
    op->checked_type_ = func->checked_type();
    return std::move(GetRef<GlobalVar>(op));
  }

  CallValues SchemaToValue(Array<Expr> args, const Op op) {
    CallValues call_values = CallValues::make();
    Array<Value> arg_values;
    for (const auto& arg : args) {
      if (var_value_map_.count(arg.as<VarNode>())) {
        arg_values.push_back(GetValue(var_value_map_[arg.as<VarNode>()]));
      } else {
        arg_values.push_back(GetValue(arg));
      }
    }
    call_values->args = GetOpAttr<op::FRAFSchema>(op, "FRAFSchema")(arg_values);
    call_values->callee = OpValue::make(op);
    return call_values;
  }

  Expr VisitExpr_(const CallNode* call) override {
    static const Op& invoke_op = Op::Get("raf.op.vm.invoke_op");
    const OpNode* opn = call->op.as<OpNode>();
    if (opn && GetRef<Op>(opn) == invoke_op) {
      // Since invoke_op use the second argument (input tuple) to invoke
      // the first argument (op or closure), we need to update the var map
      // of input tuple (args) to the closure. Just like %closure(%input_var).
      auto fn_var = call->args[0].as<VarNode>();
      CHECK(call->args[1].as<VarNode>())
          << "The 2nd argument of invoke_op must be tuple, but got " << call->args[1]->GetTypeKey();
      auto input_var = call->args[1].as<VarNode>();
      CHECK(var_value_map_.count(input_var) != 0)
          << "The 2nd argument of invoke_op hasn't be visited";
      const auto input_tuple = Downcast<Tuple>(var_value_map_[input_var]);
      VisitPrimitiveClosureFromCallerArgs(fn_var, input_tuple->fields);
    }

    Array<Expr> args;
    for (const auto& arg : call->args) {
      args.push_back(VisitExpr(arg));
    }

    static const auto declare_op = Op::GetAttrMap<op::FRAFDeclare>("FRAFDeclare");
    // We do constant-folding for shape-related operators by invoking their declare function,
    // because they produce shape information which is required by type inference.
    // The arguments (SchemaToValue(args)) passed to declare function
    // can be either types or tensor values, depends on whether
    // they have already been evaluated/constant-folded.
    // Therefore it is essential to deal with both cases in their declare functions.

    // TODO(@hgt312): refactor concatenate_dx be a base op and only use types
    static std::unordered_set<std::string> shape_list{
        "raf.op.shape", "raf.op.get_reduce_axis", "raf.op.get_kept_dims", "raf.op.concatenate_dx"};
    if (opn && shape_list.count(opn->name)) {
      CallValues call_values = SchemaToValue(args, GetRef<Op>(opn));
      declare_op[GetRef<Op>(opn)](call_values);
      if (call_values->out.defined()) {
        Expr re = ir::MakeConstant(call_values->out);
        return VisitExpr(re);
      }
    }

    if (const FunctionNode* fn = call->op.as<FunctionNode>()) {
      UpdateFuncParamVarMap(fn, call->args);
    }

    Call ret = Call(call->op, args, call->attrs, call->type_args);
    if (const FunctionNode* fn = call->op.as<FunctionNode>()) {
      auto ret_type = InferClosure(ret, GetRef<Function>(fn));
      ret = Call(VisitExpr(call->op), args, call->attrs, call->type_args);
      ret->checked_type_ = ret_type;
    } else if (const GlobalVarNode* gvn = call->op.as<GlobalVarNode>()) {
      auto fn = Downcast<Function>(mod_->Lookup(GetRef<GlobalVar>(gvn)));
      auto ret_type = InferClosure(ret, fn);
      ret = Call(VisitExpr(call->op), args, call->attrs, call->type_args);
      ret->op->checked_type_ = Unify(gvn->checked_type(), fn->checked_type());
      ret->checked_type_ = ret_type;
    } else {
      Expr op = VisitExpr(call->op);
      ret = Call(op, args, call->attrs, call->type_args);
      if (const OpNode* opn = ret->op.as<OpNode>()) {
        ret->checked_type_ = InferPrimitive(ret, GetRef<Op>(opn));
      } else if (ret->op.as<VarNode>() || ret->op.as<LetNode>()) {
        // handle recursive func call when op is a var node
        if (op->checked_type()->IsInstance<IncompleteTypeNode>()) {
          ret->checked_type_ = IncompleteType(kType);
        } else {
          // The var node can be a result of the output type of a func call. A var node
          // here is valid if it points to a function. Check that the type is a FuncType
          // and the args of the Call match the type of the FuncType. If yes, return the
          // FuncType's ret_type.
          if (const auto* var_node = ret->op.as<VarNode>()) {
            VisitPrimitiveClosureFromCallerArgs(var_node, call->args);
          }
          const FuncTypeNode* fty_node = ret->op->checked_type_.as<FuncTypeNode>();
          CHECK(fty_node);
          for (size_t i = 0; i < fty_node->arg_types.size(); i++) {
            ret->args[i]->checked_type_ =
                Unify(fty_node->arg_types[i], ret->args[i]->checked_type());
          }
          ret->checked_type_ = fty_node->ret_type;
        }
      } else if (const auto* ftn = op->checked_type().as<FuncTypeNode>()) {
        ret->checked_type_ = ftn->ret_type;
      } else {
        LOG(FATAL) << "Invalid op type: " << call->op->GetTypeKey();
      }
    }
    return ret;
  }

  Type InferPrimitive(const Call& call, const Op op) {
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
    try {
      return ti->func(call_values);
    } catch (const dmlc::Error& e) {
      LOG(FATAL) << "Failed to infer type of the following primitive: " << std::endl
                 << raf::ir::AsText(call) << std::endl
                 << std::endl
                 << e.what();
    }
  }

  Type InferClosure(const Call& call, const Function& fn) {
    // TODO(@hzfan): perform template param deduction to eliminate type_params
    bool update_closure = false;
    Function curr_fn = fn;
    if (visited_closures_.count(fn) > 0) {
      curr_fn = visited_closures_[fn];
    }

    Array<Var> new_params;
    for (size_t i = 0; i < call->args.size(); ++i) {
      try {
        // Try to unify caller type and param type.
        Unify(call->args[i]->checked_type(), curr_fn->params[i]->type_annotation);
        new_params.push_back(
            MakeVar(curr_fn->params[i]->name_hint(), curr_fn->params[i]->type_annotation));
      } catch (const dmlc::Error& e) {
        // If caller type and closure parameter type are inconsistent and this is the first caller,
        // update the closure parameter type; othewise throw an error.
        CHECK(visited_closures_.find(curr_fn) == visited_closures_.end())
            << "The following closure is called more than once "
            << "but callers have inconsistent types:" << std::endl
            << raf::ir::AsText(curr_fn) << std::endl
            << e.what();
        update_closure = true;
        new_params.push_back(
            MakeVar(curr_fn->params[i]->name_hint(), call->args[i]->checked_type()));
      }
    }

    if (update_closure) {
      // If param types have to be updated, create a new closure with updated param types.
      // Note that in this case we also have to mutate the closure body to use the updated
      // param vars, so the closure body cannot be visited in advance.
      for (size_t i = 0; i < new_params.size(); ++i) {
        closure_param_map_[curr_fn->params[i]] = new_params[i];
      }
      curr_fn = WithFields(curr_fn, new_params);
      UpdateFuncParamVarMap(curr_fn.as<FunctionNode>(), call->args);
    }
    curr_fn = Downcast<Function>(VisitExpr(curr_fn));

    // Mark both the original and updated closure as visited because they are not allowed
    // to be updated anymore.
    visited_closures_[fn] = curr_fn;
    visited_closures_[curr_fn] = curr_fn;
    return Downcast<FuncType>(curr_fn->checked_type())->ret_type;
  }

  void UpdateFuncParamVarMap(const FunctionNode* fn, const Array<Expr>& args) {
    // Map the function parameters in var_value_map_ to the caller arguments in order to
    // infer type of function body. As a result, this has to be done before visiting the
    // function body.
    CHECK_EQ(args.size(), fn->params.size());
    for (size_t n = args.size(), i = 0; i < n; ++i) {
      Expr arg = VisitExpr(args[i]);
      const auto* v = arg.as<VarNode>();
      if (v && var_value_map_.count(v)) {
        var_value_map_[fn->params[i].get()] = var_value_map_[v];
      } else {
        var_value_map_[fn->params[i].get()] = arg;
      }
    }
  }

  void VisitPrimitiveClosureFromCallerArgs(const VarNode* fn_var, const Array<Expr>& args) {
    // The visit of binded closure has been deferred until visiting its callers.
    // When visiting callers, we use their arguments to update the closure parameters in
    // var_value_map_ and then visit the closure body.
    auto fn_expr = var_value_map_[fn_var];
    const FunctionNode* fn = fn_expr.as<FunctionNode>();
    if (!fn || !fn->HasNonzeroAttr(attr::kPrimitive)) {
      return;
    }
    UpdateFuncParamVarMap(fn, args);
    auto new_fn = VisitExpr(GetRef<Function>(fn));
    fn_var->checked_type_ = new_fn->checked_type();
    var_value_map_[fn_var] = new_fn;
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
      op->checked_type_ = VoidType();
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
    auto pre_visit = [this](const LetNode* op) {
      Expr ovalue = op->value;
      Var var = op->var;
      Expr value = ovalue;

      // Do not infer binded primitive functions here. Since we may need the caller arguments
      // to infer types of function body, we defer type inference of primitive function to its
      // caller.
      auto fn_node = value.as<FunctionNode>();
      bool infer_body = !fn_node || !fn_node->HasNonzeroAttr(attr::kPrimitive);
      if (infer_body) {
        value = VisitExpr(ovalue);
      }

      if (value.as<ConstantNode>()) {
        this->memo_[var] = value;
        return;
      }

      const VarNode* v = value.as<VarNode>();
      if (v && var_value_map_.count(v)) {
        var_value_map_[op->var.get()] = var_value_map_[v];
      } else {
        var_value_map_[op->var.get()] = value;
      }

      // If the binded primitive function has not been inferred, then it does not have the type yet.
      if (infer_body) {
        var->checked_type_ = value->checked_type();
      }
    };
    auto post_visit = [this](const LetNode* op) {
      auto expr = GetRef<Expr>(op);
      Expr ovalue = op->value;
      Var var = op->var;
      Expr value = ovalue;

      // Do not infer binded primitive functions here. Since we may need the caller arguments
      // to infer types of function body, we defer type inference of primitive function to its
      // caller.
      auto fn_node = value.as<FunctionNode>();
      bool infer_body = !fn_node || !fn_node->HasNonzeroAttr(attr::kPrimitive);
      if (infer_body) {
        value = this->VisitExpr(ovalue);
      }

      if (value.as<ConstantNode>()) {
        this->memo_[expr] = this->VisitExpr(op->body);
        return;
      }
      Expr body = this->VisitExpr(op->body);
      Let let(var, value, body);
      let->checked_type_ = body->checked_type();
      this->memo_[expr] = let;
    };
    ExpandANormalForm(op, pre_visit, post_visit);
    return memo_[GetRef<Expr>(op)];
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

  Expr VisitExpr_(const OpNode* node) override {
    auto op = GetRef<Op>(node);
    op->checked_type_ = GetOpAttr<OpType>(op, "OpType");
    return op;
  }

  Expr VisitExpr_(const FunctionNode* op) override {
    auto fn = GetRef<Function>(op);
    if (visited_closures_.count(fn) > 0) {
      fn = visited_closures_[fn];
    }
    if (visited_.count(fn)) {
      if (!fn->checked_type_.defined()) {
        fn->checked_type_ = IncompleteType(kType);
      }
      return fn;
    }
    visited_.insert(fn);
    Array<Var> params;
    Array<Type> param_types;
    for (const auto& p : fn->params) {
      Var param = Downcast<Var>(VisitExpr(p));
      params.push_back(param);
      param_types.push_back(param->checked_type());
    }
    Expr body = VisitExpr(fn->body);
    Type ret_type =
        fn->ret_type.defined() ? Unify(body->checked_type(), fn->ret_type) : body->checked_type();
    Function func(params, body, ret_type, fn->type_params, fn->attrs);
    func->checked_type_ = FuncType(param_types, ret_type, fn->type_params, {});
    return func;
  }

 private:
  IRModule mod_;
  /*! \brief The var_value_map_ is used to track Let binding Expr.
   * E.g. Let %a = %b; Let %c = some_op(%a). The var_value_map_ will map %b to some_op.
   */
  std::unordered_map<const VarNode*, Expr> var_value_map_;
  /*! \brief Mapping from original closures to visited ones (may have type-updated params). */
  std::unordered_map<Function, Function, ObjectPtrHash, ObjectPtrEqual> visited_closures_;
  /*! \brief Mapping from original closure params to type-updated ones. */
  std::unordered_map<Var, Var, ObjectPtrHash, ObjectPtrEqual> closure_param_map_;
  /*! \brief Track visited Expr to avoid indefinite recursion in IR with recursive functions */
  std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual> visited_;
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

    auto left_index = lhs.as<tvm::IntImmNode>();
    auto right_index = rhs.as<tvm::IntImmNode>();
    if (!left_index && right_index) {
      return lhs;
    } else if (left_index && !right_index) {
      return rhs;
    } else if (left_index && right_index && left_index->value == right_index->value) {
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
    CHECK(tt1->dtype == tt2->dtype) << "dtype mismatch: " << tt1->dtype << " vs. " << tt2->dtype;

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

Pass InferType() {
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

Expr InferType(Expr func) {
  auto mod = GlobalModule();
  return type_infer::TypeInferencer(mod).VisitExpr(func);
}

Expr InferTypeWithValues(const Expr& func, const Array<Value>& values) {
  auto mod = GlobalModule();
  auto ti = type_infer::TypeInferencer(mod);
  Array<Expr> args;
  for (const auto& v : values) {
    // It's safe to wrap the values with `MakeConstant`, as the type functions
    // need to access the real data in values, and there is no write operation.
    args.push_back(MakeConstant(v));
  }
  ti.UpdateFuncParamVarMap(func.as<FunctionNode>(), args);
  return ti.VisitExpr(func);
}

Expr InferTypeWithModule(const Expr& expr, const IRModule& m) {
  IRModule mod(m->functions, m->type_definitions, m->Imports());
  int idx = 0;
  std::string gv_name;
  do {
    std::ostringstream oss;
    oss << "_tmp" << idx;
    gv_name = oss.str();
    ++idx;
  } while (mod->ContainGlobalVar(gv_name));
  GlobalVar gvar(gv_name);
  BaseFunc func;
  if (expr.as<FunctionNode>()) {
    func = Downcast<Function>(expr);
  } else {
    func = Function(pass::FreeVars(expr), expr, Type(), pass::FreeTypeVars(expr, mod), {});
  }
  mod->Add(gvar, func);
  mod = InferType()(mod);
  Expr ret;
  if (expr.as<FunctionNode>()) {
    ret = mod->Lookup(gvar);
  } else {
    ret = mod->Lookup(gvar).as<FunctionNode>()->body;
  }
  return ret;
}

RAF_REGISTER_GLOBAL("raf.pass_.InferType").set_body_typed([]() { return InferType(); });

}  // namespace pass
}  // namespace raf
