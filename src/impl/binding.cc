/*!
 * Copyright (c) 2019 by Contributors
 * \file src/impl/binding.cc
 * \brief Frontend-defined varioble-expression-value bindings
 */
#include "mnm/binding.h"
#include "mnm/registry.h"
#include "mnm/op.h"
#include "mnm/executor.h"
#include "mnm/tensor.h"

namespace mnm {
namespace binding {

using namespace mnm::ir;
using namespace mnm::value;
using executor::interpreter::InvokeClosure;
using op::CallValues;
using op::MakeListArgs;
using registry::GetPackedFunc;
using tensor::Tensor;

class BindingMgr {
 public:
  std::mutex mu;
  std::unordered_map<const VarNode*, BindingEntry> bindings;

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
    BindingEntry entry{nullptr};
    {
      std::lock_guard<std::mutex> lock(mgr->mu);
      auto iter = mgr->bindings.find(this);
      CHECK(iter != mgr->bindings.end());
      entry = iter->second;
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

GradTape GradTape::make(Var grad, ClosureValue bp, Array<ObjectRef> prev_tapes) {
  ObjectPtr<GradTapeObj> n = make_object<GradTapeObj>();
  n->grad = std::move(grad);
  n->bp = std::move(bp);
  n->prev_tapes = std::move(prev_tapes);
  return GradTape(n);
}

NDArrayBinding NDArrayBinding::make(Value value, GradTape tape) {
  ObjectPtr<NDArrayBindingObj> n = make_object<NDArrayBindingObj>();
  n->value = std::move(value);
  n->tape = std::move(tape);
  return NDArrayBinding(n);
}

SymbolBinding SymbolBinding::make(Expr expr) {
  ObjectPtr<SymbolBindingObj> n = make_object<SymbolBindingObj>();
  n->expr = std::move(expr);
  return SymbolBinding(n);
}

Var MakeManagedBinding(const BindingEntry& entry, const std::string& name_hint) {
  static BindingMgr* mgr = BindingMgr::Get();
  static auto& bindings = mgr->bindings;
  Var var = BoundVarObj::make(name_hint);
  const VarNode* var_ptr = var.operator->();
  {
    std::lock_guard<std::mutex> lock(mgr->mu);
    bindings.emplace(var_ptr, entry);
  }
  return var;
}

Var BindNDArray(Value value, GradTape tape, std::string name_hint) {
  std::string grad_name_hint = "d" + name_hint;
  return MakeManagedBinding(NDArrayBinding::make(
                                /*value=*/std::move(value),
                                /*tape=*/tape),
                            name_hint);
}

void RebindNDArray(Var var, Value value, GradTape tape) {
  static BindingMgr* mgr = BindingMgr::Get();
  static auto& bindings = mgr->bindings;
  {
    std::lock_guard<std::mutex> lock(mgr->mu);
    auto iter = bindings.find(var.operator->());
    CHECK(iter != bindings.end()) << "Rebind var does not exist!";
    iter->second = NDArrayBinding::make(value, tape);
  }
}

Var BindSymbol(Expr expr, std::string name_hint) {
  return MakeManagedBinding(SymbolBinding::make(std::move(expr)), name_hint);
}

BindingEntry LookupBinding(const VarNode* var) {
  static BindingMgr* mgr = BindingMgr::Get();
  static const auto& bindings = mgr->bindings;
  {
    std::lock_guard<std::mutex> lock(mgr->mu);
    auto iter = bindings.find(var);
    return iter != bindings.end() ? iter->second : NullValue<BindingEntry>();
  }
}

Value LookupBoundValue(Var var) {
  return Downcast<NDArrayBinding>(LookupBinding(var.operator->()))->value;
}

ObjectRef DeTuple(Value value) {
  if (value->IsInstance<ScalarValueObj>() || value->IsInstance<ClosureValueObj>()) {
    return std::move(value);
  }
  if (value->IsInstance<TensorValueObj>()) {
    return BindNDArray(value);
  }
  if (const auto* tuple = value.as<TupleValueObj>()) {
    Array<ObjectRef> result;
    int n = static_cast<int>(tuple->fields.size());
    for (int i = 0; i < n; ++i) {
      Value sub_value = tuple->fields[i];
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

ObjectRef DeStruct(Value value, ClosureValue bp, Array<ObjectRef> prev_tapes) {
  if (value->IsInstance<ScalarValueObj>()) {
    return std::move(value);
  }
  GradTape tape = GradTape::make(
      /*dy=*/binding::BindNDArray({}),
      /*bp=*/std::move(bp),
      /*prev_tapes=*/std::move(prev_tapes));
  if (value->IsInstance<TensorValueObj>()) {
    return BindNDArray(std::move(value), std::move(tape));
  }
  if (const auto* tuple = value.as<TupleValueObj>()) {
    Array<ObjectRef> result;
    int n = static_cast<int>(tuple->fields.size());
    Var dy = VarNode::make("dy", {});
    std::vector<Expr> grads(n, MakeConstant(NoGradValue::make()));
    for (int i = 0; i < n; ++i) {
      Value sub_value = tuple->fields[i];
      if (sub_value->op_env == nullptr) {
        sub_value->op_env = tuple->op_env;
      }
      grads[i] = dy;
      result.push_back(DeStruct(
          /*value=*/sub_value,
          /*bp=*/ClosureValue::make({}, FunctionNode::make({dy}, TupleNode::make(grads), {}, {})),
          /*prev_tapes*/ {tape}));
    }
    return std::move(result);
  }
  LOG(FATAL) << "ValueError: cannot de-tuple " << value->GetTypeKey();
  throw;
}

void SetRequiresGrad(Var var, bool value) {
  GradTape& tape = Downcast<NDArrayBinding>(LookupBinding(var.operator->()))->tape;
  if (tape.defined() == value) {
    return;
  }
  if (value) {
    tape = GradTape::make(BindNDArray({}, {}, "d" + var->name_hint()), {}, {});
  } else {
    tape = NullValue<GradTape>();
  }
}

Var LookupGrad(Var var) {
  GradTape& tape = Downcast<NDArrayBinding>(LookupBinding(var.operator->()))->tape;
  return tape.defined() ? tape->grad : NullValue<Var>();
}

TensorValue MakeOnes(Context to_ctx) {
  static float a[1] = {1.0};
  static int64_t b[1] = {1};
  DType dtype = DType(DTypeCode::kFloat(), 32, 1);
  DLTensor tensor;
  tensor.data = a;
  tensor.ctx = Context(DevType::kCPU(), 0);
  tensor.dtype = dtype;
  tensor.shape = b;
  tensor.ndim = 0;
  tensor.strides = nullptr;
  tensor.byte_offset = 0;
  auto array = tvm::runtime::NDArray::Empty({}, dtype, to_ctx);
  array.CopyFrom(&tensor);
  return TensorValue::make(Tensor::FromDLPack(array.ToDLPack()));
}

void Backward(Var var, Var dy_var) {
  Context y_ctx = Downcast<TensorValue>(LookupBoundValue(var))->tensor->ctx;
  Value dy = dy_var.defined() ? Downcast<NDArrayBinding>(LookupBinding(dy_var.operator->()))->value
                              : MakeOnes(y_ctx);
  GradTape tape = Downcast<NDArrayBinding>(LookupBinding(var.operator->()))->tape;
  if (!tape.defined()) {
    return;
  }
  Array<ObjectRef> prev_tapes = tape->prev_tapes;
  Value _dxs = InvokeClosure(CallValues::make(tape->bp, MakeListArgs({dy})));
  if (const auto* dx = _dxs.as<TensorValueObj>()) {
    CHECK_EQ(prev_tapes.size(), 1U);
    const auto* grad_tape = prev_tapes[0].as<GradTapeObj>();
    if (grad_tape && !dx->IsInstance<NoGradValueObj>()) {
      RebindNDArray(grad_tape->grad, _dxs);
    }
  } else if (const auto* dxs = _dxs.as<TupleValueObj>()) {
    CHECK_EQ(dxs->fields.size(), prev_tapes.size());
    int n = dxs->fields.size();
    for (int i = 0; i < n; ++i) {
      Value value = dxs->fields[i];
      const auto* grad_tape = prev_tapes[i].as<GradTapeObj>();
      if (grad_tape && value.defined() && !value->IsInstance<NoGradValueObj>()) {
        RebindNDArray(grad_tape->grad, std::move(value));
      }
    }
  }
}

MNM_REGISTER_GLOBAL("mnm.binding.BindNDArray").set_body_typed(BindNDArray);
MNM_REGISTER_GLOBAL("mnm.binding.BindSymbol").set_body_typed(BindSymbol);
MNM_REGISTER_GLOBAL("mnm.binding.LookupBoundValue").set_body_typed(LookupBoundValue);
MNM_REGISTER_GLOBAL("mnm.binding.SetRequiresGrad").set_body_typed(SetRequiresGrad);
MNM_REGISTER_GLOBAL("mnm.binding.Backward").set_body_typed(Backward);
MNM_REGISTER_GLOBAL("mnm.binding.LookupGrad").set_body_typed(LookupGrad);

namespace {
MNM_REGISTER_OBJECT_NO_REFLECT(GradTapeObj);
MNM_REGISTER_OBJECT_NO_REFLECT(BindingEntryObj);
MNM_REGISTER_OBJECT_NO_REFLECT(NDArrayBindingObj);
MNM_REGISTER_OBJECT_NO_REFLECT(SymbolBindingObj);
}  // namespace
}  // namespace binding
}  // namespace mnm
