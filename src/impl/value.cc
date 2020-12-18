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

NoGradValue NoGradValue::make() {
  ObjectPtr<NoGradValueObj> n = make_object<NoGradValueObj>();
  return NoGradValue(n);
}

VoidValue VoidValue::make() {
  ObjectPtr<VoidValueObj> n = make_object<VoidValueObj>();
  return VoidValue(n);
}

TensorTypeValue TensorTypeValue::make(TensorType t) {
  ObjectPtr<TensorTypeValueObj> n = make_object<TensorTypeValueObj>();
  n->type = std::move(t);
  return TensorTypeValue(n);
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

Value::operator tensor::Tensor &() const {
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

TensorValue TensorValue::Assemble(const Context& ctx, const DType& dtype,
                                  const Array<IntValue> shape_array,
                                  const std::vector<int64_t>& strides, void* const data) {
  std::vector<int64_t> shape;
  for (auto value : shape_array) {
    shape.push_back(value->data);
  }
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
MNM_REGISTER_GLOBAL("mnm.value._make.ClosureValue").set_body_typed(ClosureValue::make);
MNM_REGISTER_GLOBAL("mnm.value._make.NoGradValue").set_body_typed(NoGradValue::make);

MNM_REGISTER_OBJECT_NO_REFLECT(ValueObj);
MNM_REGISTER_OBJECT_NO_REFLECT(BaseTensorValueObj);
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
MNM_REGISTER_OBJECT_REFLECT(TensorTypeValueObj);
MNM_REGISTER_OBJECT_REFLECT(NoGradValueObj);
}  // namespace value
}  // namespace mnm
