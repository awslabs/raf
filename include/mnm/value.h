#pragma once

#include <dlpack/dlpack.h>

#include <mnm/base.h>
#include <mnm/ir.h>
#include <mnm/tensor.h>

namespace mnm {
namespace value {

/* Value */
class ValueNode : public ir::Node {
 public:
  static constexpr const char* _type_key = "mnm.value.Value";
  MNM_DEF_BASE_NODE_INFO(ValueNode, ir::Node);
};

class Value : public ir::NodeRef {
 public:
  operator const DLTensor*() const;
  operator const tensor::Tensor&() const;
  MNM_DEF_NODE_REF_METHODS(Value, ir::NodeRef, ValueNode);
};

/* TensorValue */
class TensorValueNode final : public ValueNode {
 public:
  tensor::Tensor tensor;
  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("tensor", &tensor);
  }
  static constexpr const char* _type_key = "mnm.value.TensorValue";
  MNM_DEF_NODE_TYPE_INFO(TensorValueNode, ValueNode);
};

class TensorValue final : public Value {
 public:
  static TensorValue make(tensor::Tensor tensor);
  static TensorValue Assemble(const Context& ctx, const DType& dtype,
                              const std::vector<int64_t>& shape,
                              const std::vector<int64_t>& strides = {}, void* data = nullptr);
  MNM_DEF_NODE_REF_METHODS(TensorValue, Value, TensorValueNode);
};

/* TupleValue */
class TupleValueNode final : public ValueNode {
 public:
  ir::Array<Value> fields;
  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("fields", &fields);
  }
  static constexpr const char* _type_key = "mnm.value.TupleValue";
  MNM_DEF_NODE_TYPE_INFO(TupleValueNode, ValueNode);
};

class TupleValue final : public Value {
 public:
  static TupleValue make(ir::Array<Value> fields);
  MNM_DEF_NODE_REF_METHODS(TupleValue, Value, TupleValueNode);
};

/* ClosureValue */
class ClosureValueNode final : public ValueNode {
 public:
  ir::Map<ir::Var, Value> env;
  ir::Function func;
  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("env", &env);
    v->Visit("func", &func);
  }
  static constexpr const char* _type_key = "mnm.value.ClosureValue";
  MNM_DEF_NODE_TYPE_INFO(ClosureValueNode, ValueNode);
};

class ClosureValue final : public Value {
 public:
  static ClosureValue make(ir::Map<ir::Var, Value> value, ir::Function func);
  MNM_DEF_NODE_REF_METHODS(ClosureValue, Value, ClosureValueNode);
};

/* RefValue */
class RefValueNode final : public ValueNode {
 public:
  mutable Value value;
  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("value", &value);
  }
  static constexpr const char* _type_key = "mnm.value.RefValue";
  MNM_DEF_NODE_TYPE_INFO(RefValueNode, ValueNode);
};

class RefValue final : public Value {
 public:
  static RefValue make(Value value);
  MNM_DEF_NODE_REF_METHODS(RefValue, Value, RefValueNode);
};

/* ConstructorValue */
class ConstructorValueNode;
class ConstructorValue;

/* OpaqueValue */
class OpaqueValueNode : public ValueNode {
 public:
  mutable ir::NodeRef data{nullptr};
  static constexpr const char* _type_key = "mnm.value.OpaqueValue";
  MNM_DEF_NODE_TYPE_INFO(OpaqueValueNode, ValueNode);
};

class OpaqueValue : public Value {
 public:
  MNM_DEF_NODE_REF_METHODS(OpaqueValue, Value, OpaqueValueNode);
};

}  // namespace value
}  // namespace mnm
