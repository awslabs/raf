#pragma once

#include <dlpack/dlpack.h>

#include <mnm/base.h>
#include <mnm/rly.h>
#include <mnm/tensor.h>

namespace mnm {
namespace value {

/* Value */
class ValueNode : public rly::Node {
 public:
  static constexpr const char* _type_key = "mnm.value.Value";
  MNM_DEF_BASE_NODE_INFO(ValueNode, rly::Node);
};

class Value : public rly::NodeRef {
 public:
  MNM_DEF_NODE_REF_METHODS(Value, rly::NodeRef, ValueNode);
  operator const DLTensor*() const;
  operator const tensor::Tensor&() const;
};

/* OpaqueValue */
class OpaqueValueNode : public ValueNode {
 public:
  static constexpr const char* _type_key = "mnm.value.OpaqueValue";
  MNM_DEF_BASE_NODE_INFO(OpaqueValueNode, ValueNode);
};

class OpaqueValue : public Value {
 public:
  MNM_DEF_NODE_REF_METHODS(OpaqueValue, Value, ValueNode);
};

/* ScalarValue */
class ScalarValueNode final : public ValueNode {
 public:
  // TODO(@junrushao1994): we need both int64 and float64, probably string?
  rly::Integer data;

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

/* TensorValue */
class TensorValueNode final : public ValueNode {
 public:
  tensor::Tensor tensor;

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
  static TensorValue Assemble(Context ctx,                        //
                              DType dtype,                        //
                              std::vector<int64_t> shape,         //
                              std::vector<int64_t> strides = {},  //
                              void* data = nullptr);
};

/* TupleValue */
class TupleValueNode final : public ValueNode {
 public:
  rly::Array<Value> fields;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("fields", &fields);
  }
  static constexpr const char* _type_key = "mnm.value.TupleValue";
  MNM_DEF_NODE_TYPE_INFO(TupleValueNode, ValueNode);
};

class TupleValue final : public Value {
  MNM_DEF_NODE_REF_METHODS(TupleValue, Value, TupleValueNode);
};

/* ClosureValue */
class ClosureValue;
class ClosureValueNode;

/* RefValue */
class RefValue;
class RefValueNode;

/* ConstructorValue */
class ConstructorValue;
class ConstructorValueNode;

}  // namespace value
}  // namespace mnm
