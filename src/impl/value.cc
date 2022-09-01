/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/impl/value.cc
 * \brief RAF value underlying implementation
 */
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/node/functor.h>
#include <tvm/ir/module.h>
#include "raf/executor.h"
#include "raf/ir_ext.h"
#include "raf/registry.h"
#include "raf/tensor.h"
#include "raf/value.h"
#include "../common/shape_utils.h"

#ifdef RAF_USE_CUDA
#include "../../src/common/cuda_utils.h"
#include "../../src/op/dialect/cudnn/cudnn_utils.h"
#include "../../src/op/dialect/cublas/cublas_utils.h"
#endif

namespace raf {
namespace value {

using common::shape_utils::GetShape;
using common::shape_utils::MakeShape;
using executor::Executor;
using tensor::Tensor;
using namespace raf::ir;

ValueType TypeKey2ValueType(const char* type_key) {
  if (strcmp(type_key, "raf.value.IntValue") == 0) {
    return kIntValue;
  }
  if (strcmp(type_key, "raf.value.FloatValue") == 0) {
    return kFloatValue;
  }
  if (strcmp(type_key, "raf.value.BoolValue") == 0) {
    return kBoolValue;
  }
  if (strcmp(type_key, "raf.value.StringValue") == 0) {
    return kStringValue;
  }
  if (strcmp(type_key, "raf.value.TensorValue") == 0) {
    return kTensorValue;
  }
  if (strcmp(type_key, "raf.value.TenorTypeValue") == 0) {
    return kTensorTypeValue;
  }
  if (strcmp(type_key, "raf.value.TupleValue") == 0) {
    return kTupleValue;
  }
  if (strcmp(type_key, "raf.value.ClosureValue") == 0) {
    return kClosureValue;
  }
  if (strcmp(type_key, "raf.value.RefValue") == 0) {
    return kRefValue;
  }
  if (strcmp(type_key, "raf.value.OpValue") == 0) {
    return kOpValue;
  }
  if (strcmp(type_key, "raf.value.OpaqueValue") == 0) {
    return kOpaqueValue;
  }
  if (strcmp(type_key, "raf.value.NoGradValue") == 0) {
    return kNoGradValue;
  }
  if (strcmp(type_key, "raf.value.VoidValue") == 0) {
    return kVoidValue;
  }
  LOG(FATAL) << "Unknown value type key: " << type_key;
  return kNullptr;
}

std::string ValueType2String(ValueType type) {
  switch (type) {
    case kNullptr:
      return "nullptr";
    case kIntValue:
      return "IntValue";
    case kFloatValue:
      return "FloatValue";
    case kBoolValue:
      return "BoolValue";
    case kStringValue:
      return "StringValue";
    case kTensorValue:
      return "TensorValue";
    case kTensorTypeValue:
      return "TensorTypeValue";
    case kTupleValue:
      return "TupleValue";
    case kClosureValue:
      return "CloureValue";
    case kRefValue:
      return "RefValue";
    case kOpValue:
      return "OpValue";
    case kOpaqueValue:
      return "OpaqueValue";
    case kNoGradValue:
      return "NoGradValue";
    case kVoidValue:
      return "VoidValue";
    default:
      LOG(FATAL) << "Unknown value type: " << static_cast<int32_t>(type);
      return "";
  }
}

/*** Constructors ***/
TensorValue TensorValue::make(tensor::Tensor tensor, std::shared_ptr<memory_pool::Memory> mem) {
  ObjectPtr<TensorValueObj> n = make_object<TensorValueObj>();
  n->tensor = std::move(tensor);
  n->mem = std::move(mem);
  return TensorValue(n);
}

TupleValue TupleValue::make(Array<Value> fields) {
  ObjectPtr<TupleValueObj> n = make_object<TupleValueObj>();
  n->fields = std::move(fields);
  return TupleValue(n);
}

ClosureValue ClosureValue::make(Map<Var, Value> env, Function func, Optional<Var> bind) {
  ObjectPtr<ClosureValueObj> n = make_object<ClosureValueObj>();
  n->env = std::move(env);
  n->func = std::move(func);
  n->bind = std::move(bind);
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

IntValue ScalarValue::make(int8_t value) {
  return IntValue::make(DataType::Int(8), value);
}

IntValue ScalarValue::make(int16_t value) {
  return IntValue::make(DataType::Int(16), value);
}

IntValue ScalarValue::make(int32_t value) {
  return IntValue::make(DataType::Int(32), value);
}

IntValue ScalarValue::make(int64_t value) {
  return IntValue::make(DataType::Int(64), value);
}

IntValue ScalarValue::make(uint8_t value) {
  return IntValue::make(DataType::UInt(8), value);
}

IntValue ScalarValue::make(uint16_t value) {
  return IntValue::make(DataType::UInt(16), value);
}

IntValue ScalarValue::make(uint32_t value) {
  return IntValue::make(DataType::UInt(32), value);
}

IntValue ScalarValue::make(uint64_t value) {
  return IntValue::make(DataType::UInt(64), value);
}

FloatValue ScalarValue::make(float value) {
  return FloatValue::make(DataType::Float(32), value);
}

FloatValue ScalarValue::make(double value) {
  return FloatValue::make(DataType::Float(64), value);
}

BoolValue ScalarValue::make(bool value) {
  return BoolValue::make(value);
}

IntValue IntValue::make(DataType dtype, int64_t value) {
  ObjectPtr<IntValueObj> n = make_object<IntValueObj>();
  n->dtype = dtype;
  n->value = value;
  return IntValue(n);
}

FloatValue FloatValue::make(DataType dtype, double value) {
  ObjectPtr<FloatValueObj> n = make_object<FloatValueObj>();
  n->dtype = dtype;
  n->value = value;
  return FloatValue(n);
}

BoolValue BoolValue::make(bool value) {
  ObjectPtr<BoolValueObj> n = make_object<BoolValueObj>();
  n->dtype = DataType::Bool();
  n->value = value;
  return BoolValue(n);
}

StringValue StringValue::make(const std::string& value) {
  ObjectPtr<StringValueObj> n = make_object<StringValueObj>();
  n->value = value;
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
  } else if (const auto* tensor_t_value = this->as<TensorTypeValueObj>()) {
    // In this case, create a TensorValueObject out of it
    TensorType ty = tensor_t_value->type;
    auto ty_node = ty.as<TensorTypeNode>();
    CHECK(ty_node);
    std::vector<int64_t> shape;
    for (auto i : ty_node->shape) {
      if (auto dim_shape = i.as<ir::IntImmNode>())
        shape.push_back(dim_shape->value);
      else
        LOG(FATAL) << "Cannot convert to TensorValue due to dynamic shape!";
    }
    DLDataType dtype = ty_node->dtype;
    // In this case we create a tensor based on the given type, so the target device
    // must be available.
    Tensor t = Tensor::make(Device::Current(), dtype, shape);
    TensorValue tv = TensorValue::make(t);
    const DLTensor* dl_tensor_ref = tv->tensor.operator->();
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
TensorValue TensorValue::Assemble(const Device& dev, const DType& dtype,
                                  const std::vector<int64_t>& shape,
                                  const std::vector<int64_t>& strides, void* const data,
                                  std::shared_ptr<memory_pool::Memory> mem) {
  return TensorValue::make(Tensor::make(dev, dtype, shape, strides, data), std::move(mem));
}

TensorValue TensorValue::Assemble(const Device& dev, const DType& dtype,
                                  const Array<IntValue> shape_array,
                                  const std::vector<int64_t>& strides, void* const data,
                                  std::shared_ptr<memory_pool::Memory> mem) {
  std::vector<int64_t> shape;
  for (auto value : shape_array) {
    shape.push_back(value->value);
  }
  return TensorValue::make(Tensor::make(dev, dtype, shape, strides, data), std::move(mem));
}

TensorValue TensorValue::CreateView(const std::vector<int64_t>& shape,
                                    const std::vector<int64_t>& strides) const {
  return TensorValue::make((*this)->tensor.CreateView(shape, strides), (*this)->mem);
}

TensorValue AssembleTensorValue(const Device& dev, DLDataType dtype, Array<Integer> shape,
                                Array<Integer> strides, void* data) {
  return TensorValue::make(
      Tensor::make(dev, dtype, MakeShape<int64_t>(shape), MakeShape<int64_t>(strides), data));
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
  if (value->IsInstance<TensorValueObj>() || value->IsInstance<NoGradValueObj>()) {
    return std::move(value);
  }
  if (value->IsInstance<ScalarValueObj>()) {
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

Value CopyTo(Value src, const Device& dev) {
  if (!src.defined()) {
    return src;
  }
  if (src.as<TensorValueObj>()) {
    auto tensor = Downcast<TensorValue>(src)->tensor;
    if (tensor->device.device_type != dev.device_type()) {
      return TensorValue::make(tensor::Tensor(tensor.CopyTo(dev)));
    }
    return src;
  }
  if (src.as<TupleValueObj>()) {
    std::vector<Value> ret;
    TupleValue tup = Downcast<TupleValue>(src);
    for (size_t i = 0; i < tup->fields.size(); ++i) {
      ret.push_back(CopyTo(tup->fields[i], dev));
    }
    return TupleValue::make(ret);
  }
  return src;
}

void CopyTo(Value src, Value dst) {
  if (!src.defined()) {
    return;
  }

  if (src.as<TensorValueObj>()) {
    CHECK(dst.as<TensorValueObj>());
    auto in_tensor = Downcast<TensorValue>(src)->tensor;
    auto out_tensor = Downcast<TensorValue>(dst)->tensor;
    in_tensor.CopyTo(out_tensor);
  } else if (src.as<TupleValueObj>()) {
    CHECK(dst.as<TupleValueObj>());
    std::vector<Value> ret;
    TupleValue in_tuple = Downcast<TupleValue>(src);
    TupleValue out_tuple = Downcast<TupleValue>(dst);
    for (size_t i = 0; i < in_tuple->fields.size(); ++i) {
      CopyTo(in_tuple->fields[i], out_tuple->fields[i]);
    }
  } else {
    LOG(FATAL) << "Unrecognized value: " << src->GetTypeKey();
  }
}

Value CreateDummyValueFromType(const tvm::Type& type, Device device) {
  if (auto tensor_type = type.as<tvm::TensorTypeNode>()) {
    std::vector<int64_t> shape;
    for (auto v : tensor_type->shape) {
      const auto* int_imm = v.as<tvm::IntImmNode>();
      CHECK(int_imm != nullptr) << "Only supports creating dummy tensor value with static shape.";
      shape.push_back(tvm::Integer(GetRef<tvm::IntImm>(int_imm)).IntValue());
    }
    int64_t nbytes = tensor_type->dtype.bytes();
    for (auto v : shape) {
      nbytes *= v;
    }
    std::shared_ptr<memory_pool::Memory> memory = memory_pool::Memory::Alloc(device, nbytes);
    DLDataType data_type = tensor_type->dtype;

    // Set integer values to zeros to avoid memory errors in certain ops
    // E.g., raf.op.tvm.nll_loss would have CUDA memory errors because the class
    // index is out of range
    if ((data_type.code == kDLInt) || (data_type.code == kDLUInt)) {
#ifdef RAF_USE_CUDA
      if (device.device_type() == DevType::kCUDA())
        CUDA_CALL(cudaMemset(memory->data, 0, nbytes));
      else
#endif
        memset(memory->data, 0, nbytes);
    }
    return TensorValue::Assemble(device, data_type, shape, {}, memory->data, memory);
  } else if (auto tuple_type = type.as<tvm::TupleTypeNode>()) {
    tvm::Array<Value> fields;
    for (auto field_type : tuple_type->fields)
      fields.push_back(CreateDummyValueFromType(field_type, device));
    return TupleValue::make(fields);
  } else {
    LOG(FATAL) << "NotImplementedError: Do not support creating dummy value for type " << type;
    throw;
  }
}

RAF_REGISTER_GLOBAL("raf.value.AssembleTensorValue")
    .set_body_typed([](const tvm::Device& dev, DLDataType dtype, Array<Integer> shape,
                       Array<Integer> strides, void* data) {
      return AssembleTensorValue(Device(dev), dtype, shape, strides, data);
    });
RAF_REGISTER_GLOBAL("raf.value.DeTuple").set_body_typed(DeTuple);
RAF_REGISTER_GLOBAL("raf.value.FromTVM").set_body_typed(FromTVM);
RAF_REGISTER_GLOBAL("raf.value.ToTVM").set_body_typed(ToTVM);
RAF_REGISTER_GLOBAL("raf.value._make.TupleValue").set_body_typed(TupleValue::make);
RAF_REGISTER_GLOBAL("raf.value._make.IntValue").set_body_typed(IntValue::make);
RAF_REGISTER_GLOBAL("raf.value._make.FloatValue").set_body_typed(FloatValue::make);
RAF_REGISTER_GLOBAL("raf.value._make.BoolValue").set_body_typed(BoolValue::make);
RAF_REGISTER_GLOBAL("raf.value._make.StringValue").set_body_typed(StringValue::make);
RAF_REGISTER_GLOBAL("raf.value._make.ClosureValue").set_body_typed(ClosureValue::make);
RAF_REGISTER_GLOBAL("raf.value._make.NoGradValue").set_body_typed(NoGradValue::make);

RAF_REGISTER_OBJECT_NO_REFLECT(ValueObj);
RAF_REGISTER_OBJECT_NO_REFLECT(BaseTensorValueObj);
RAF_REGISTER_OBJECT_NO_REFLECT(ScalarValueObj);
RAF_REGISTER_OBJECT_NO_REFLECT(OpaqueValueObj);

RAF_REGISTER_OBJECT_REFLECT(TensorValueObj);
RAF_REGISTER_OBJECT_REFLECT(TupleValueObj);
RAF_REGISTER_OBJECT_REFLECT(ClosureValueObj);
RAF_REGISTER_OBJECT_REFLECT(RefValueObj);
RAF_REGISTER_OBJECT_REFLECT(OpValueObj);
RAF_REGISTER_OBJECT_REFLECT(IntValueObj);
RAF_REGISTER_OBJECT_REFLECT(FloatValueObj);
RAF_REGISTER_OBJECT_REFLECT(BoolValueObj);
RAF_REGISTER_OBJECT_REFLECT(StringValueObj);
RAF_REGISTER_OBJECT_REFLECT(TensorTypeValueObj);
RAF_REGISTER_OBJECT_REFLECT(NoGradValueObj);

using tvm::ReprPrinter;
using tvm::runtime::ObjectRef;
TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TupleValueObj>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const TupleValueObj*>(ref.get());
      p->stream << "TupleValue(" << node->fields << ")";
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<IntValueObj>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const IntValueObj*>(ref.get());
      p->stream << node->dtype << "(" << node->value << ")";
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<FloatValueObj>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const FloatValueObj*>(ref.get());
      p->stream << node->dtype << "(" << node->value << ")";
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<BoolValueObj>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const BoolValueObj*>(ref.get());
      p->stream << "bool(" << node->value << ")";
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<StringValueObj>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const StringValueObj*>(ref.get());
      p->stream << "str\"" << node->value << "\"";
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TensorValueObj>([](const ObjectRef& ref, ReprPrinter* p) {
      static auto* dev2str = tvm::runtime::Registry::Get("raf._core.core_utils.dev2str");
      auto* node = static_cast<const TensorValueObj*>(ref.get());
      p->stream << "tensor(";
      for (int i = 0; i < node->tensor->ndim; ++i) {
        p->stream << node->tensor->shape[i];
        if (i != node->tensor->ndim - 1) p->stream << "x";
      }
      p->stream << ", " << tvm::runtime::DLDataType2String(node->tensor->dtype);
      tvm::String device_str = (*dev2str)(node->tensor->device);
      p->stream << ", " << device_str << ")";
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<OpValueObj>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const OpValueObj*>(ref.get());
      p->stream << node->op;
    });

}  // namespace value
}  // namespace raf
