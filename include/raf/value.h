/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file value.h
 * \brief Definition of RAF values
 */
#pragma once
#include <tvm/relay/type.h>
#include <memory>
#include <vector>
#include <string>
#include "./ir.h"
#include "./tensor.h"
#include "./memory_pool.h"
#include "../3rdparty/tvm/src/relay/transforms/pattern_utils.h"

namespace raf {
namespace op {
class OpEnv;
}  // namespace op
}  // namespace raf

// Basic values used in tensor algebra
namespace raf {
namespace value {

/* Value type */
enum ValueType {
  kNullptr = 0,
  kIntValue = 1,
  kFloatValue = 2,
  kBoolValue = 3,
  kStringValue = 4,
  kTensorValue = 5,
  kTensorTypeValue = 6,
  kTupleValue = 7,
  kClosureValue = 8,
  kRefValue = 9,
  kOpValue = 10,
  kOpaqueValue = 11,
  kNoGradValue = 12,
  kVoidValue = 13,
};

/*!
 * \brief Convert value type key to value type.
 * \param type_key The type key.
 * \return The corresponding value type.
 */
ValueType TypeKey2ValueType(const char* type_key);

/*!
 * \brief Get the name of the value type.
 * \param type The value type.
 * \return The name.
 */
std::string ValueType2String(ValueType type);

/* Value */
class ValueObj : public ir::Object {
 public:
  mutable std::shared_ptr<op::OpEnv> op_env{nullptr};
  static constexpr const uint32_t _type_index = ir::TypeIndex::kDynamic;
  static constexpr const char* _type_key = "raf.value.Value";
  RAF_BASE_OBJECT(ValueObj, ir::Object);
};

class Value : public ir::ObjectRef {
 public:
  operator DLTensor*() const;
  operator tensor::Tensor &() const;
  template <typename TValue,
            typename = typename std::enable_if<std::is_base_of<Value, TValue>::value>::type>
  explicit operator TValue() const {
    return ir::Downcast<TValue>(*this);
  }
  RAF_OBJECT_REF(Value, ir::ObjectRef, ValueObj);
};

// Scalar values

class IntValue;
class FloatValue;
class BoolValue;

/* ScalarValue */
class ScalarValueObj : public ValueObj {
 public:
  ir::DataType dtype;
  static constexpr const uint32_t _type_index = tvm::TypeIndex::kDynamic;
  static constexpr const char* _type_key = "raf.value.ScalarValue";
  RAF_BASE_OBJECT(ScalarValueObj, ValueObj);
};

class ScalarValue : public Value {
 public:
  static IntValue make(int8_t value);
  static IntValue make(int16_t value);
  static IntValue make(int32_t value);
  static IntValue make(int64_t value);
  static IntValue make(uint8_t value);
  static IntValue make(uint16_t value);
  static IntValue make(uint32_t value);
  static IntValue make(uint64_t value);
  static FloatValue make(float value);
  static FloatValue make(double value);
  static BoolValue make(bool value);
  RAF_OBJECT_REF(ScalarValue, Value, ScalarValueObj);
};

/* IntValue */
class IntValueObj : public ScalarValueObj {
 public:
  int64_t value;
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("dtype", &dtype);
    v->Visit("value", &value);
  }
  bool SEqualReduce(const IntValueObj* other, tvm::SEqualReducer equal) const {
    return equal(dtype, other->dtype) && equal(value, other->value);
  }
  void SHashReduce(tvm::SHashReducer hash_reduce) const {
    hash_reduce(dtype);
    hash_reduce(value);
  }
  static constexpr const char* _type_key = "raf.value.IntValue";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  RAF_FINAL_OBJECT(IntValueObj, ScalarValueObj);
};

class IntValue : public ScalarValue {
 public:
  static IntValue make(ir::DataType dtype, int64_t value);
  RAF_OBJECT_REF(IntValue, ScalarValue, IntValueObj);
};

/* FloatValue */
class FloatValueObj : public ScalarValueObj {
 public:
  double value;
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("dtype", &dtype);
    v->Visit("value", &value);
  }
  bool SEqualReduce(const FloatValueObj* other, tvm::SEqualReducer equal) const {
    return equal(dtype, other->dtype) && equal(value, other->value);
  }
  void SHashReduce(tvm::SHashReducer hash_reduce) const {
    hash_reduce(dtype);
    hash_reduce(value);
  }
  static constexpr const char* _type_key = "raf.value.FloatValue";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  RAF_FINAL_OBJECT(FloatValueObj, ScalarValueObj);
};

class FloatValue : public ScalarValue {
 public:
  static FloatValue make(ir::DataType dtype, double value);
  RAF_OBJECT_REF(FloatValue, ScalarValue, FloatValueObj);
};

/* BoolValue */
class BoolValueObj : public ScalarValueObj {
 public:
  bool value;
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("dtype", &dtype);
    v->Visit("value", &value);
  }
  bool SEqualReduce(const BoolValueObj* other, tvm::SEqualReducer equal) const {
    return equal(dtype, other->dtype) && equal(value, other->value);
  }
  void SHashReduce(tvm::SHashReducer hash_reduce) const {
    hash_reduce(dtype);
    hash_reduce(value);
  }
  static constexpr const char* _type_key = "raf.value.BoolValue";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  RAF_FINAL_OBJECT(BoolValueObj, ScalarValueObj);
};

class BoolValue : public ScalarValue {
 public:
  static BoolValue make(bool data);
  RAF_OBJECT_REF(BoolValue, ScalarValue, BoolValueObj);
};

/* BaseTensorValue */
class BaseTensorValueObj : public ValueObj {
 public:
  static constexpr const uint32_t _type_index = ir::TypeIndex::kDynamic;
  static constexpr const char* _type_key = "raf.value.BaseTensorValue";
  RAF_BASE_OBJECT(BaseTensorValueObj, ValueObj);
};

class BaseTensorValue : public Value {
 public:
  RAF_OBJECT_REF(BaseTensorValue, Value, BaseTensorValueObj);
};

/* TensorValue */
class TensorValueObj final : public BaseTensorValueObj {
 public:
  mutable tensor::Tensor tensor;
  mutable std::shared_ptr<memory_pool::Memory> mem{nullptr};
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("_tensor", &tensor);
  }
  bool SEqualReduce(const TensorValueObj* other, tvm::SEqualReducer equal) const {
    // TODO(@hgt312): pointer equal now
    return this == other;
  }
  void SHashReduce(tvm::SHashReducer hash_reduce) const {
    // TODO(@hgt312): pointer equal now
    const void* ptr = reinterpret_cast<const void*>(this);
    hash_reduce->SHashReduceHashedValue(std::hash<const void*>()(ptr));
  }
  static constexpr const char* _type_key = "raf.value.TensorValue";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  RAF_FINAL_OBJECT(TensorValueObj, BaseTensorValueObj);
};

class TensorValue final : public BaseTensorValue {
 public:
  static TensorValue make(tensor::Tensor tensor,
                          std::shared_ptr<memory_pool::Memory> mem = nullptr);
  static TensorValue Assemble(const Device& dev, const DType& dtype,
                              const std::vector<int64_t>& shape,
                              const std::vector<int64_t>& strides = {}, void* data = nullptr,
                              std::shared_ptr<memory_pool::Memory> mem = nullptr);
  static TensorValue Assemble(const Device& dev, const DType& dtype,
                              const ir::Array<IntValue> shape,
                              const std::vector<int64_t>& strides = {}, void* data = nullptr,
                              std::shared_ptr<memory_pool::Memory> mem = nullptr);
  TensorValue CreateView(const std::vector<int64_t>& shape = {},
                         const std::vector<int64_t>& strides = {}) const;
  RAF_OBJECT_REF(TensorValue, BaseTensorValue, TensorValueObj);
};

/* TensorTypeValue */
class TensorTypeValueObj final : public BaseTensorValueObj {
 public:
  tvm::relay::TensorType type;
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("_type", &type);
  }
  bool SEqualReduce(const TensorTypeValueObj* other, tvm::SEqualReducer equal) const {
    return equal(type->shape, other->type->shape) && equal(type->dtype, other->type->dtype);
  }
  void SHashReduce(tvm::SHashReducer hash_reduce) const {
    hash_reduce(type->shape);
    hash_reduce(type->dtype);
  }
  static constexpr const char* _type_key = "raf.value.TensorTypeValue";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  RAF_FINAL_OBJECT(TensorTypeValueObj, BaseTensorValueObj);
};

class TensorTypeValue final : public BaseTensorValue {
 public:
  static TensorTypeValue make(tvm::relay::TensorType type);
  RAF_OBJECT_REF(TensorTypeValue, BaseTensorValue, TensorTypeValueObj);
};

/* TupleValue */
class TupleValueObj final : public ValueObj {
 public:
  ir::Array<Value> fields;
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("_fields", &fields);
  }
  bool SEqualReduce(const TupleValueObj* other, tvm::SEqualReducer equal) const {
    // Treat empty tuple as a constant node instead of a graph node.
    if (fields.size() == other->fields.size() && fields.size() == 0) {
      return true;
    } else {
      equal->MarkGraphNode();
      return equal(fields, other->fields);
    }
  }
  void SHashReduce(tvm::SHashReducer hash_reduce) const {
    if (fields.size() != 0) {
      hash_reduce->MarkGraphNode();
      for (size_t i = 0; i < fields.size(); ++i) {
        hash_reduce(fields[i]);
      }
    }
  }
  static constexpr const char* _type_key = "raf.value.TupleValue";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  RAF_FINAL_OBJECT(TupleValueObj, ValueObj);
};

class TupleValue final : public Value {
 public:
  static TupleValue make(ir::Array<Value> fields);
  RAF_OBJECT_REF(TupleValue, Value, TupleValueObj);
};

/* ClosureValue */
class ClosureValueObj final : public ValueObj {
 public:
  /*! \brief The set of free variables in the closure.
   *
   * These are the captured variables which are required for
   * evaluation when we call the closure.
   */
  ir::Map<ir::Var, Value> env;
  /*! \brief The function which implements the closure.
   *
   * \note May reference the variables contained in the env.
   */
  ir::Function func;
  /*! \brief variable the closure bind to, used when
   * the function is a recursive function.
   */
  ir::Optional<ir::Var> bind;
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("_env", &env);
    v->Visit("_func", &func);
  }
  static constexpr const char* _type_key = "raf.value.ClosureValue";
  RAF_FINAL_OBJECT(ClosureValueObj, ValueObj);
};

class ClosureValue final : public Value {
 public:
  static ClosureValue make(ir::Map<ir::Var, Value> env, ir::Function func,
                           ir::Optional<ir::Var> bind = tvm::NullOpt);
  RAF_OBJECT_REF(ClosureValue, Value, ClosureValueObj);
};

/* RefValue */
class RefValueObj final : public ValueObj {
 public:
  mutable Value value;
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("_value", &value);
  }
  static constexpr const char* _type_key = "raf.value.RefValue";
  RAF_FINAL_OBJECT(RefValueObj, ValueObj);
};

class RefValue final : public Value {
 public:
  static RefValue make(Value value);
  RAF_OBJECT_REF(RefValue, Value, RefValueObj);
};

/* OpValue */
class OpValueObj final : public ValueObj {
 public:
  ir::Op op;
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("_op", &op);
  }
  static constexpr const char* _type_key = "raf.value.OpValue";
  RAF_FINAL_OBJECT(OpValueObj, ValueObj);
};

class OpValue final : public Value {
 public:
  static OpValue make(ir::Op op);
  RAF_OBJECT_REF(OpValue, Value, OpValueObj);
};

/* ConstructorValue */
class ConstructorValueObj;
class ConstructorValue;

/* OpaqueValue */
class OpaqueValueObj : public ValueObj {
 public:
  mutable Value data{nullptr};
  static constexpr const char* _type_key = "raf.value.OpaqueValue";
  RAF_FINAL_OBJECT(OpaqueValueObj, ValueObj);
};

class OpaqueValue : public Value {
 public:
  RAF_OBJECT_REF(OpaqueValue, Value, OpaqueValueObj);
};

/* StringValue */
class StringValueObj : public ValueObj {
 public:
  std::string value;
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("value", &value);
  }
  bool SEqualReduce(const StringValueObj* other, tvm::SEqualReducer equal) const {
    return equal(value, other->value);
  }
  void SHashReduce(tvm::SHashReducer hash_reduce) const {
    hash_reduce(value);
  }
  static constexpr const char* _type_key = "raf.value.StringValue";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  RAF_FINAL_OBJECT(StringValueObj, ValueObj);
};

class StringValue : public Value {
 public:
  static StringValue make(const std::string& data);
  RAF_OBJECT_REF(StringValue, Value, StringValueObj);
};

/* Specific values */
class NoGradValueObj : public ValueObj {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) {
  }
  bool SEqualReduce(const NoGradValueObj* other, tvm::SEqualReducer equal) const {
    return true;
  }
  void SHashReduce(tvm::SHashReducer hash_reduce) const {
  }
  static constexpr const char* _type_key = "raf.value.NoGradValue";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  RAF_FINAL_OBJECT(NoGradValueObj, ValueObj);
};

class NoGradValue : public Value {
 public:
  static NoGradValue make();
  RAF_OBJECT_REF(NoGradValue, Value, NoGradValueObj);
};

/* Null values */
class VoidValueObj : public ValueObj {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) {
  }
  static constexpr const char* _type_key = "raf.value.VoidValue";
  RAF_FINAL_OBJECT(VoidValueObj, ValueObj);
};

class VoidValue : public Value {
 public:
  static VoidValue make();
  RAF_OBJECT_REF(VoidValue, Value, VoidValueObj);
};

template <typename T>
T GetScalarValueData(const Value& value) {
  using namespace tvm::runtime;

  if (const auto* bvo = value.as<BoolValueObj>()) {
    return bvo->value;
  } else if (const auto* fvo = value.as<FloatValueObj>()) {
    return fvo->value;
  } else if (const auto* ivo = value.as<IntValueObj>()) {
    return ivo->value;
  } else if (const auto* tvo = value.as<TensorValueObj>()) {
    tensor::Tensor tensor = tvo->tensor;
    CHECK(tensor->ndim == 0U || (tensor->ndim == 1U && tensor->shape[0] == 1))
        << "Value is not a scalar";

    DataType dtype = DataType(tensor->dtype);
    NDArray nd_array;
    if (tensor->device.device_type != kDLCPU) {
      DLDevice cpu_dev;
      cpu_dev.device_type = kDLCPU;
      cpu_dev.device_id = 0;
      nd_array = tensor.CopyTo(cpu_dev);
    } else {
      nd_array = tensor;
    }
    void* raw_data = nd_array->data;

    T data;
    TVM_DTYPE_DISPATCH(dtype, DType, {
      if (dtype == DataType::Float(16)) {
        // The storage of float16 is uint16_t. Here we convert it to float32.
        data = __extendXfYf2__<uint16_t, uint16_t, 10, float, uint32_t, 23>(
            reinterpret_cast<uint16_t*>(raw_data)[0]);
      } else if (dtype == DataType::Bool()) {
        data = reinterpret_cast<uint8_t*>(raw_data)[0];
      } else {
        data = static_cast<DType*>(raw_data)[0];
      }
    });
    return data;
  }
  LOG(FATAL) << "Cannot convert to scalar value";
}

/*!
 * \brief Copy a value to specified device.
 * \param src Value to be copyed.
 * \param dev The pecified device.
 * \return The copyed value.
 */
Value CopyTo(Value src, const Device& dev);

/*!
 * \brief Copy a value to another value.
 * \param src Value to be copyed.
 * \param dst The destination.
 */
void CopyTo(Value src, Value dst);

/*!
 * \brief Create a dummy value according to type.
 * \param type The tensor or tuple type.
 * \param device The device to allocate memory from when creating a tensor.
 * \return The dummy value.
 */
Value CreateDummyValueFromType(const tvm::Type& type, Device device);

}  // namespace value
}  // namespace raf
