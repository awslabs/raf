/*!
 * Copyright (c) 2019 by Contributors
 * \file src/impl/value.cc
 * \brief MNM value underlying implementation
 */
#include "tvm/runtime/ndarray.h"
#include "mnm/executor.h"
#include "mnm/ir.h"
#include "mnm/registry.h"
#include "mnm/tensor.h"
#include "mnm/value.h"
#include "../common/shape_utils.h"

namespace mnm {
namespace value {

using common::shape_utils::GetShape;
using common::shape_utils::MakeShape;
using executor::Executor;
using tensor::Tensor;
using namespace mnm::ir;

/*** Constructors ***/
TensorValue TensorValue::make(tensor::Tensor tensor) {
  ObjectPtr<TensorValueObj> n = make_object<TensorValueObj>();
  n->tensor = std::move(tensor);
  return TensorValue(n);
}

TupleValue TupleValue::make(Array<Value> fields) {
  ObjectPtr<TupleValueObj> n = make_object<TupleValueObj>();
  n->fields = std::move(fields);
  return TupleValue(n);
}

ClosureValue ClosureValue::make(Map<Var, Value> env, Function func) {
  ObjectPtr<ClosureValueObj> n = make_object<ClosureValueObj>();
  n->env = std::move(env);
  n->func = std::move(func);
  return ClosureValue(n);
}

RefValue RefValue::make(Value value) {
  ObjectPtr<RefValueObj> n = make_object<RefValueObj>();
  n->value = std::move(value);
  return RefValue(n);
}

OpValue OpValue::make(Op op) {
  ObjectPtr<OpValueObj> n = make_object<OpValueObj>();
  n->op = std::move(op);
  return OpValue(n);
}

IntValue ScalarValue::make(int data) {
  return IntValue::make(data);
}

IntValue ScalarValue::make(int64_t data) {
  return IntValue::make(data);
}

FloatValue ScalarValue::make(double data) {
  return FloatValue::make(data);
}

BoolValue ScalarValue::make(bool data) {
  return BoolValue::make(data);
}

IntValue IntValue::make(int64_t data) {
  ObjectPtr<IntValueObj> n = make_object<IntValueObj>();
  n->data = data;
  return IntValue(n);
}

FloatValue FloatValue::make(double data) {
  ObjectPtr<FloatValueObj> n = make_object<FloatValueObj>();
  n->data = data;
  return FloatValue(n);
}

BoolValue BoolValue::make(bool data) {
  ObjectPtr<BoolValueObj> n = make_object<BoolValueObj>();
  n->data = data;
  return BoolValue(n);
}

StringValue StringValue::make(const std::string& data) {
  ObjectPtr<StringValueObj> n = make_object<StringValueObj>();
  n->data = data;
  return StringValue(n);
}

/*** GetType ***/
Type GetType(const Value& value) {
  if (const auto* tv = value.as<TensorValueObj>()) {
    const DLTensor& dlt = *tv->tensor.operator->();
    auto shape = GetShape<tvm::Integer>(dlt);
    return ir::TensorTypeNode::make({shape.begin(), shape.end()}, tvm::TVMType2Type(dlt.dtype));
  } else if (const auto* tv = value.as<TupleValueObj>()) {
    Array<Type> tuple_type;
    for (const Value& sub_value : tv->fields) {
      tuple_type.push_back(GetType(sub_value));
    }
    return ir::TupleTypeNode::make(tuple_type);
  }
  LOG(FATAL) << "NotImplementedError: " << value->GetTypeKey();
  throw;
}

/*** Value ***/
Value::operator DLTensor*() const {
  if (const auto* tensor_value = this->as<TensorValueObj>()) {
    const DLTensor* dl_tensor_ref = tensor_value->tensor.operator->();
    return const_cast<DLTensor*>(dl_tensor_ref);
  }
  LOG(FATAL) << "InternalError: cannot convert to TensorValue";
  throw;
}

Value::operator tensor::Tensor&() const {
  if (const auto* tensor_value = this->as<TensorValueObj>()) {
    return tensor_value->tensor;
  }
  LOG(FATAL) << "InternalError: cannot convert to TensorValue";
  throw;
}

/*** TensorValue ***/
TensorValue TensorValue::Assemble(const Context& ctx, const DType& dtype,
                                  const std::vector<int64_t>& shape,
                                  const std::vector<int64_t>& strides, void* const data) {
  return TensorValue::make(Tensor::make(ctx, dtype, shape, strides, data));
}

TensorValue AssembleTensorValue(DLContext ctx, DLDataType dtype, Array<Integer> shape,
                                Array<Integer> strides, void* data) {
  return TensorValue::make(
      Tensor::make(ctx, dtype, MakeShape<int64_t>(shape), MakeShape<int64_t>(strides), data));
}

TensorValue FromTVM(tvm::runtime::NDArray array) {
  return TensorValue::make(Tensor::FromDLPack(array.ToDLPack()));
}

/*** External symbols ***/
tvm::runtime::NDArray ToTVM(TensorValue value) {
  DLManagedTensor* tensor = value->tensor.ToDLPack();
  if (tensor->dl_tensor.strides != nullptr) {
    tensor->deleter(tensor);
    LOG(FATAL) << "NotImplementedError: strided tensor not supported";
    throw;
  }
  return tvm::runtime::NDArray::FromDLPack(tensor);
}

ObjectRef DeTuple(Value value) {
  if (value->IsInstance<TensorValueObj>()) {
    return std::move(value);
  }
  if (const auto* tuple = value.as<TupleValueObj>()) {
    Array<ObjectRef> result;
    for (Value sub_value : tuple->fields) {
      if (sub_value->op_env == nullptr) {
        sub_value->op_env = tuple->op_env;
      }
      result.push_back(DeTuple(sub_value));
    }
    return std::move(result);
  }
  LOG(FATAL) << "ValueError: cannot de-tuple " << value->GetTypeKey();
  throw;
}

MNM_REGISTER_GLOBAL("mnm.value.AssembleTensorValue").set_body_typed(AssembleTensorValue);
MNM_REGISTER_GLOBAL("mnm.value.DeTuple").set_body_typed(DeTuple);
MNM_REGISTER_GLOBAL("mnm.value.FromTVM").set_body_typed(FromTVM);
MNM_REGISTER_GLOBAL("mnm.value.ToTVM").set_body_typed(ToTVM);
MNM_REGISTER_GLOBAL("mnm.value._make.TupleValue").set_body_typed(TupleValue::make);
MNM_REGISTER_GLOBAL("mnm.value._make.IntValue").set_body_typed(IntValue::make);
MNM_REGISTER_GLOBAL("mnm.value._make.FloatValue").set_body_typed(FloatValue::make);
MNM_REGISTER_GLOBAL("mnm.value._make.BoolValue").set_body_typed(BoolValue::make);
MNM_REGISTER_GLOBAL("mnm.value._make.StringValue").set_body_typed(StringValue::make);
MNM_REGISTER_OBJECT_NO_REFLECT(ValueObj);
MNM_REGISTER_OBJECT_NO_REFLECT(ScalarValueObj);
MNM_REGISTER_OBJECT_NO_REFLECT(OpaqueValueObj);
MNM_REGISTER_OBJECT_REFLECT(TensorValueObj);
MNM_REGISTER_OBJECT_REFLECT(TupleValueObj);
MNM_REGISTER_OBJECT_REFLECT(ClosureValueObj);
MNM_REGISTER_OBJECT_REFLECT(RefValueObj);
MNM_REGISTER_OBJECT_REFLECT(OpValueObj);
MNM_REGISTER_OBJECT_REFLECT(IntValueObj);
MNM_REGISTER_OBJECT_REFLECT(FloatValueObj);
MNM_REGISTER_OBJECT_REFLECT(BoolValueObj);
MNM_REGISTER_OBJECT_REFLECT(StringValueObj);
}  // namespace value
}  // namespace mnm

namespace mnm {
namespace value {

class BindingEntry {
 public:
  Expr expr{nullptr};
  Value value{nullptr};

  BindingEntry() = default;
  BindingEntry(const Expr& expr, const Value& value) : expr(expr), value(value) {
  }
};

class BindingMgr {
 public:
  std::mutex mu;
  std::unordered_map<const VarNode*, std::unique_ptr<BindingEntry> > bindings;

  static BindingMgr* Get() {
    static BindingMgr* instance = new BindingMgr();
    return instance;
  }
};

class BoundVarObj : public VarNode {
  // This is basically relay::VarNode, but with a customized callback that
  // deletes the weak reference inside BindingMgr
 public:
  ~BoundVarObj() {
    static BindingMgr* mgr = BindingMgr::Get();
    std::unique_ptr<BindingEntry> entry{nullptr};
    {
      std::lock_guard<std::mutex> lock(mgr->mu);
      auto iter = mgr->bindings.find(this);
      CHECK(iter != mgr->bindings.end());
      entry.swap(iter->second);
      mgr->bindings.erase(iter);
    }
    // "entry" is destroyed here, to avoid potential recursive lock
  }
  static Var make(const std::string& name_hint) {
    ObjectPtr<BoundVarObj> n = make_object<BoundVarObj>();
    ObjectPtr<IdNode> id_ptr = make_object<IdNode>();
    id_ptr->name_hint = name_hint;
    n->vid = Id(id_ptr);
    return Var(n);
  }
};

Var BindNothing(const std::string& name_hint) {
  return BindExprValue(NullValue<Expr>(), NullValue<Value>(), name_hint);
}

Var BindValue(const Value& value, const std::string& name_hint) {
  return BindExprValue(MakeConstant(value), value, name_hint);
}

Var BindExpr(const Expr& expr, const std::string& name_hint) {
  return BindExprValue(expr, NullValue<Value>(), name_hint);
}

Var BindExprValue(const Expr& expr, const Value& value, const std::string& name_hint) {
  static BindingMgr* mgr = BindingMgr::Get();
  static auto& bindings = mgr->bindings;
  Var var = BoundVarObj::make(name_hint);
  const VarNode* var_ptr = var.operator->();
  std::unique_ptr<BindingEntry> entry = std::make_unique<BindingEntry>(expr, value);
  {
    std::lock_guard<std::mutex> lock(mgr->mu);
    bindings.emplace(var_ptr, std::move(entry));
  }
  return var;
}

Expr _LookupBoundExpr(const VarNode* var) {
  static BindingMgr* mgr = BindingMgr::Get();
  static const auto& bindings = mgr->bindings;
  {
    std::lock_guard<std::mutex> lock(mgr->mu);
    auto iter = bindings.find(var);
    return iter != bindings.end() ? iter->second->expr : NullValue<Expr>();
  }
}

Value _LookupBoundValue(const VarNode* var) {
  static BindingMgr* mgr = BindingMgr::Get();
  static const auto& bindings = mgr->bindings;
  {
    std::lock_guard<std::mutex> lock(mgr->mu);
    auto iter = bindings.find(var);
    return iter != bindings.end() ? iter->second->value : NullValue<Value>();
  }
}

Expr LookupBoundExpr(const Var& var) {
  return _LookupBoundExpr(var.operator->());
}

Value LookupBoundValue(const ir::Var& var) {
  return _LookupBoundValue(var.operator->());
}

class LetListExtractor final : public ExprVisitor {
 public:
  void AddVar(const Expr& expr) {
    if (const VarNode* var = expr.as<VarNode>()) {
      if (++in_degree[var] == 1) {
        queue.push_back(var);
      }
      if (out_edge != nullptr) {
        out_edge->push_back(var);
      }
    } else if (!expr->IsInstance<ConstantNode>()) {
      LOG(FATAL) << "Every intermediate result should be bound to a relay.Var";
    }
  }

  void VisitExpr_(const VarNode* var) final {
    if (++in_degree[var] == 1) {
      queue.push_back(var);
    }
  }

  void VisitExpr_(const TupleNode* node) final {
    for (const Expr& expr : node->fields) {
      AddVar(expr);
    }
  }

  void VisitExpr_(const CallNode* node) final {
    for (const Expr& expr : node->args) {
      AddVar(expr);
    }
  }

  void VisitExpr_(const TupleGetItemNode* node) final {
    AddVar(node->tuple);
  }

  std::vector<const VarNode*> queue;
  std::unordered_map<const VarNode*, int> in_degree;
  std::unordered_map<const VarNode*, const ExprNode*> bindings;
  std::unordered_map<const VarNode*, std::vector<const VarNode*> > graph;
  std::vector<const VarNode*>* out_edge = nullptr;

  Expr Run(const Var& var) {
    AddVar(var);
    while (!queue.empty()) {
      const VarNode* var = queue.back();
      const Expr& expr = _LookupBoundExpr(var);
      bindings[var] = expr.operator->();
      queue.pop_back();
      out_edge = &graph[var];
      if (expr.defined()) {
        ExprVisitor::VisitExpr(expr);
      }
    }
    Expr body = var;
    queue.clear();
    queue.push_back(var.operator->());
    while (!queue.empty()) {
      const VarNode* var = queue.back();
      const ExprNode* expr = bindings[var];
      queue.pop_back();
      if (expr != nullptr) {
        body = LetNode::make(GetRef<Var>(var), GetRef<Expr>(expr), body);
      }
      for (const VarNode* out : graph[var]) {
        if (--in_degree[out] == 0) {
          queue.push_back(out);
        }
      }
    }
    return body;
  }

  static Expr Extract(const Var& var) {
    std::unique_ptr<LetListExtractor> self = std::make_unique<LetListExtractor>();
    return self->Run(var);
  }
};

ir::Expr ExtractLetList(const Var& var) {
  return LetListExtractor::Extract(var);
}

MNM_REGISTER_GLOBAL("mnm.value.BindNothing").set_body_typed(BindNothing);
MNM_REGISTER_GLOBAL("mnm.value.BindValue").set_body_typed(BindValue);
MNM_REGISTER_GLOBAL("mnm.value.BindExpr").set_body_typed(BindExpr);
MNM_REGISTER_GLOBAL("mnm.value.BindExprValue").set_body_typed(BindExprValue);
MNM_REGISTER_GLOBAL("mnm.value.LookupBoundExpr").set_body_typed(LookupBoundExpr);
MNM_REGISTER_GLOBAL("mnm.value.LookupBoundValue").set_body_typed(LookupBoundValue);
MNM_REGISTER_GLOBAL("mnm.value.ExtractLetList").set_body_typed(ExtractLetList);

}  // namespace value
}  // namespace mnm
