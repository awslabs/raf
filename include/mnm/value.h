#pragma once

#include <dlpack/dlpack.h>

#include <mnm/rly.h>
#include <mnm/tensor.h>

namespace mnm {
namespace value {

/* mnm::value::Value */
class ValueNode : public mnm::rly::Node {
 public:
  static constexpr const char* _type_key = "mnm.value.Value";
  MNM_DEF_BASE_NODE_INFO(ValueNode, mnm::rly::Node);
};

class Value : public mnm::rly::NodeRef {
 public:
  MNM_DEF_NODE_REF_METHODS(Value, mnm::rly::NodeRef, ValueNode);
  inline operator const DLTensor*() const;
  inline operator const mnm::tensor::Tensor&() const;
};

/* mnm::value::OpaqueValue, used for saving extra opaque states. */
class OpaqueValueNode : public ValueNode {
 public:
  static constexpr const char* _type_key = "mnm.value.OpaqueValue";
  MNM_DEF_BASE_NODE_INFO(OpaqueValueNode, ValueNode);
};

class OpaqueValue : public Value {
 public:
  MNM_DEF_NODE_REF_METHODS(OpaqueValue, Value, ValueNode);
};

/* mnm::value::ScalarValue */
class ScalarValueNode final : public ValueNode {
 public:
  // TODO(@junrushao1994): we need both int64 and float64, probably string?
  mnm::rly::Integer data;

  ScalarValueNode() = default;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("data", &data);
  }

  static constexpr const char* _type_key = "mnm.value.ScalarValue";
  MNM_DEF_NODE_TYPE_INFO(ScalarValueNode, ValueNode);
};

class ScalarValue final : public Value {
 public:
  MNM_DEF_NODE_REF_METHODS(ScalarValue, Value, ScalarValueNode);
};

/* mnm::value::TensorValue */
class TensorValueNode final : public ValueNode {
 public:
  mnm::tensor::Tensor tensor;

  TensorValueNode() = default;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("tensor", &tensor);
  }

  static constexpr const char* _type_key = "mnm.value.TensorValue";
  MNM_DEF_NODE_TYPE_INFO(TensorValueNode, ValueNode);
};

class TensorValue final : public Value {
 public:
  MNM_DEF_NODE_REF_METHODS(TensorValue, Value, TensorValueNode);
  static TensorValue Assemble(mnm::types::Context ctx,            //
                              mnm::types::DType dtype,            //
                              std::vector<int64_t> shape,         //
                              std::vector<int64_t> strides = {},  //
                              void* data = nullptr);
};

class TupleValueNode final : public ValueNode {
 public:
  mnm::rly::Array<Value> fields;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("fields", &fields);
  }
  static constexpr const char* _type_key = "mnm.value.TupleValue";
  MNM_DEF_NODE_TYPE_INFO(TupleValueNode, ValueNode);
};

class TupleValue final : public Value {
  MNM_DEF_NODE_REF_METHODS(TupleValue, Value, TupleValueNode);
};

/* mnm::value::ClosureValue */
class ClosureValue;
class ClosureValueNode;

/* mnm::value::RefValue */
class RefValue;
class RefValueNode;

/* mnm::value::ConstructorValue */
class ConstructorValue;
class ConstructorValueNode;

inline Value::operator const DLTensor*() const {
  if (auto tensor_value = this->as<TensorValueNode>()) {
    const DLTensor* dl_tensor_ref = tensor_value->tensor.operator->();
    return dl_tensor_ref;
  }
  LOG(FATAL) << "InternalError: cannot convert to TensorValue";
  throw;
}

inline Value::operator const mnm::tensor::Tensor&() const {
  if (const auto* tensor_value = this->as<TensorValueNode>()) {
    return tensor_value->tensor;
  }
  LOG(FATAL) << "InternalError: cannot convert to TensorValue";
  throw;
}

}  // namespace value
}  // namespace mnm
