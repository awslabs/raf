/*!
 * Copyright (c) 2019 by Contributors
 * \file src/impl/value.cc
 * \brief MNM value underlying implementation
 */
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/node/functor.h>
#include <tvm/ir/module.h>
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

ValueType TypeKey2ValueType(const char* type_key) {
  if (strcmp(type_key, "mnm.value.IntValue") == 0) {
    return kIntValue;
  }
  if (strcmp(type_key, "mnm.value.FloatValue") == 0) {
    return kFloatValue;
  }
  if (strcmp(type_key, "mnm.value.BoolValue") == 0) {
    return kBoolValue;
  }
  if (strcmp(type_key, "mnm.value.StringValue") == 0) {
    return kStringValue;
  }
  if (strcmp(type_key, "mnm.value.TensorValue") == 0) {
    return kTensorValue;
  }
  if (strcmp(type_key, "mnm.value.TenorTypeValue") == 0) {
    return kTensorTypeValue;
  }
  if (strcmp(type_key, "mnm.value.TupleValue") == 0) {
    return kTupleValue;
  }
  if (strcmp(type_key, "mnm.value.ClosureValue") == 0) {
    return kClosureValue;
  }
  if (strcmp(type_key, "mnm.value.RefValue") == 0) {
    return kRefValue;
  }
  if (strcmp(type_key, "mnm.value.OpValue") == 0) {
    return kOpValue;
  }
  if (strcmp(type_key, "mnm.value.OpaqueValue") == 0) {
    return kOpaqueValue;
  }
  if (strcmp(type_key, "mnm.value.NoGradValue") == 0) {
    return kNoGradValue;
  }
  if (strcmp(type_key, "mnm.value.VoidValue") == 0) {
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

template <>
bool GetScalarValueData<bool>(const Value& value) {
  using namespace tvm::runtime;
  if (const auto* bvo = value.as<BoolValueObj>()) {
    return bvo->value;
  } else if (const auto* tvo = value.as<TensorValueObj>()) {
    tensor::Tensor tensor = tvo->tensor;
    DLDevice cpu_ctx;
    cpu_ctx.device_type = kDLCPU;
    cpu_ctx.device_id = 0;
    NDArray cpu_array = tensor.CopyTo(cpu_ctx);
    CHECK_EQ(DataType(cpu_array->dtype), DataType::Bool());
    CHECK_EQ(cpu_array->ndim, 0);
    return reinterpret_cast<uint8_t*>(cpu_array->data)[0];
  }
  LOG(FATAL) << "Cannot convert " << value->GetTypeKey() << " to scalar bool.";
}

template <>
float GetScalarValueData<float>(const Value& value) {
  using namespace tvm::runtime;
  if (const auto* fvo = value.as<FloatValueObj>()) {
    return fvo->value;
  } else if (const auto* tvo = value.as<TensorValueObj>()) {
    tensor::Tensor tensor = tvo->tensor;
    DLDevice cpu_ctx;
    cpu_ctx.device_type = kDLCPU;
    cpu_ctx.device_id = 0;
    NDArray cpu_array = tensor.CopyTo(cpu_ctx);
    CHECK(DataType(cpu_array->dtype) == DataType::Float(32) ||
          DataType(cpu_array->dtype) == DataType::Float(16));
    CHECK_EQ(cpu_array->ndim, 0);
    return reinterpret_cast<float*>(cpu_array->data)[0];
  }
  LOG(FATAL) << "Cannot convert " << value->GetTypeKey() << " to scalar float.";
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

Value CreateDummyValueFromType(const tvm::Type& type, Device device) {
  if (auto tensor_type = type.as<tvm::TensorTypeNode>()) {
    std::vector<int64_t> shape;
    for (auto v : tensor_type->shape) {
      const auto* int_imm = v.as<tvm::IntImmNode>();
      CHECK(int_imm != nullptr) << "Only supports creating dummy tensor value with static shape.";
      shape.push_back(tvm::Integer(GetRef<tvm::IntImm>(int_imm)));
    }
    int64_t nbytes = tensor_type->dtype.bytes();
    for (auto v : shape) {
      nbytes *= v;
    }
    std::shared_ptr<memory_pool::Memory> memory = memory_pool::Memory::Alloc(device, nbytes);
    DLDataType data_type = tensor_type->dtype;
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

MNM_REGISTER_GLOBAL("mnm.value.AssembleTensorValue")
    .set_body_typed([](const tvm::Device& dev, DLDataType dtype, Array<Integer> shape,
                       Array<Integer> strides, void* data) {
      return AssembleTensorValue(Device(dev), dtype, shape, strides, data);
    });
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
      static auto* dev2str = tvm::runtime::Registry::Get("mnm._core.core_utils.dev2str");
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
}  // namespace mnm
