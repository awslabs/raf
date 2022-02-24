/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/impl/binding.cc
 * \brief Frontend-defined varioble-expression-value bindings
 */
#include "raf/binding.h"
#include "raf/registry.h"
#include "raf/op.h"
#include "raf/executor.h"
#include "raf/tensor.h"
#include "../op/ty/utils.h"
#include "raf/pass.h"

namespace raf {
namespace binding {

using namespace raf::ir;
using namespace raf::value;
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

class BoundVarObj : public ExtendedVarNode {
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
  static Var make(const std::string& name_hint, Type type = Type()) {
    ObjectPtr<BoundVarObj> n = make_object<BoundVarObj>();
    ObjectPtr<IdNode> id_ptr = make_object<IdNode>();
    id_ptr->name_hint = name_hint;
    n->vid = Id(id_ptr);
    n->type_annotation = type;
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

Var MakeManagedBinding(const BindingEntry& entry, const std::string& name_hint,
                       Type type = Type()) {
  static BindingMgr* mgr = BindingMgr::Get();
  static auto& bindings = mgr->bindings;
  Var var = BoundVarObj::make(name_hint, type);
  const VarNode* var_ptr = var.operator->();
  {
    std::lock_guard<std::mutex> lock(mgr->mu);
    bindings.emplace(var_ptr, entry);
  }
  return var;
}

Var BindNDArray(Value value, GradTape tape, std::string name_hint) {
  std::string grad_name_hint = "d" + name_hint;
  Type type = op::GetType(value);
  return MakeManagedBinding(NDArrayBinding::make(
                                /*value=*/value,
                                /*tape=*/tape),
                            name_hint, type);
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

Var BindSymbol(Expr expr, std::string name_hint, Type ty) {
  return MakeManagedBinding(SymbolBinding::make(std::move(expr)), name_hint, ty);
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

Expr LookupBoundExpr(Var var) {
  return Downcast<SymbolBinding>(LookupBinding(var.operator->()))->expr;
}

ObjectRef DeTuple(Value value) {
  if (value->IsInstance<ScalarValueObj>() || value->IsInstance<ClosureValueObj>() ||
      value->IsInstance<NoGradValueObj>()) {
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

TensorValue MakeZeros(Device to_dev, std::vector<int64_t>& shape) {
  int64_t size = 1;
  for (const int64_t& elem : shape) {
    size *= elem;
  }
  std::vector<float> a(size, 0.0);
  DType dtype = DType(DTypeCode::kFloat(), 32, 1);
  DLTensor tensor;
  tensor.data = a.data();
  tensor.device = Device(DevType::kCPU(), 0);
  tensor.dtype = dtype;
  tensor.shape = shape.data();
  tensor.ndim = shape.size();
  tensor.strides = nullptr;
  tensor.byte_offset = 0;
  auto array = tvm::runtime::NDArray::Empty(shape, dtype, to_dev);
  array.CopyFrom(&tensor);
  return TensorValue::make(Tensor::FromDLPack(array.ToDLPack()));
}

Expr MakeZeros(Value value) {
  if (value->IsInstance<ScalarValueObj>()) {
    return MakeConstant(ScalarValue::make(0.0));
  } else if (const auto* tensor = value.as<TensorValueObj>()) {
    const Tensor& a = tensor->tensor;
    Device x_dev = a->device;
    std::vector<int64_t> shape(a->shape, a->shape + a->ndim);
    return MakeConstant(MakeZeros(x_dev, shape));
  } else if (const auto* tuple = value.as<TupleValueObj>()) {
    int n = static_cast<int>(tuple->fields.size());
    std::vector<Expr> zeros(n);
    for (int i = 0; i < n; ++i) {
      Value sub_value = tuple->fields[i];
      zeros[i] = MakeZeros(sub_value);
    }
    return Tuple(zeros);
  } else {
    LOG(FATAL) << "ValueError: cannot de-tuple " << value->GetTypeKey();
    throw;
  }
}

ObjectRef DeStruct(Value value, ClosureValue bp, Array<ObjectRef> prev_tapes) {
  if (value->IsInstance<ScalarValueObj>()) {
    return std::move(value);
  }
  GradTape tape = GradTape::make(
      /*dy=*/binding::BindNDArray({}),
      /*bp=*/bp,
      /*prev_tapes=*/prev_tapes);
  if (value->IsInstance<TensorValueObj>()) {
    return BindNDArray(std::move(value), std::move(tape));
  }
  if (const auto* tuple = value.as<TupleValueObj>()) {
    Array<ObjectRef> result;
    int n = static_cast<int>(tuple->fields.size());
    Var dy = raf::ir::Var("dy", {});
    Map<ir::Var, Value> env;
    Var tuple_bp = raf::ir::Var("tuple_bp", {});
    env.Set(tuple_bp, bp);
    for (int i = 0; i < n; ++i) {
      Value sub_value = tuple->fields[i];
      std::vector<Expr> grads(n);
      for (int j = 0; j < n; ++j) {  // Set the remaining parts zero.
        if (j != i) {
          grads[j] = MakeZeros(tuple->fields[j]);
        }
      }
      if (sub_value->op_env == nullptr) {
        sub_value->op_env = tuple->op_env;
      }
      grads[i] = dy;
      result.push_back(DeStruct(
          /*value=*/sub_value,
          /*bp=*/ClosureValue::make(env, Function({dy}, Call(tuple_bp, {Tuple(grads)}), {}, {})),
          /*prev_tapes*/ prev_tapes));
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

TensorValue MakeOnes(Device to_dev, DType dtype) {
  static float a[1] = {1.0};
  static int64_t b[1] = {1};
  DLTensor tensor;
  tensor.data = a;
  tensor.device = Device(DevType::kCPU(), 0);
  tensor.dtype = dtype;
  tensor.shape = b;
  tensor.ndim = 0;
  tensor.strides = nullptr;
  tensor.byte_offset = 0;
  auto array = tvm::runtime::NDArray::Empty({}, dtype, to_dev);
  array.CopyFrom(&tensor);
  return TensorValue::make(Tensor::FromDLPack(array.ToDLPack()));
}

void Backward(Var var, Var dy_var) {
  auto y_tensor = Downcast<TensorValue>(LookupBoundValue(var))->tensor;
  Device y_dev = y_tensor->device;
  DType y_dtype = y_tensor->dtype;
  Value dy = dy_var.defined() ? Downcast<NDArrayBinding>(LookupBinding(dy_var.operator->()))->value
                              : MakeOnes(y_dev, y_dtype);
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

RAF_REGISTER_GLOBAL("raf.binding.BindNDArray").set_body_typed(BindNDArray);
RAF_REGISTER_GLOBAL("raf.binding.BindSymbol").set_body_typed(BindSymbol);
RAF_REGISTER_GLOBAL("raf.binding.RebindNDArray").set_body_typed(RebindNDArray);
RAF_REGISTER_GLOBAL("raf.binding.LookupBoundValue").set_body_typed(LookupBoundValue);
RAF_REGISTER_GLOBAL("raf.binding.LookupBoundExpr").set_body_typed(LookupBoundExpr);
RAF_REGISTER_GLOBAL("raf.binding.SetRequiresGrad").set_body_typed(SetRequiresGrad);
RAF_REGISTER_GLOBAL("raf.binding.Backward").set_body_typed(Backward);
RAF_REGISTER_GLOBAL("raf.binding.LookupGrad").set_body_typed(LookupGrad);

namespace {
RAF_REGISTER_OBJECT_NO_REFLECT(GradTapeObj);
RAF_REGISTER_OBJECT_NO_REFLECT(BindingEntryObj);
RAF_REGISTER_OBJECT_NO_REFLECT(NDArrayBindingObj);
RAF_REGISTER_OBJECT_NO_REFLECT(SymbolBindingObj);
}  // namespace
}  // namespace binding
}  // namespace raf
