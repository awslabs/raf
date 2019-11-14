#pragma once

#include <mnm/ir.h>
#include <mnm/tensor.h>

namespace mnm {
namespace op {
class OpEnv;
}  // namespace op
namespace executor {
class Executor;
}  // namespace executor
}  // namespace mnm

// Basic values used in tensor algebra
namespace mnm {
namespace value {

/* Value */
class ValueNode : public ir::Node {
 public:
  mutable std::shared_ptr<op::OpEnv> op_env{nullptr};
  static constexpr const char* _type_key = "mnm.value.Value";
  MNM_DEF_BASE_NODE_INFO(ValueNode, ir::Node);
};

class Value : public ir::NodeRef {
 public:
  operator DLTensor*() const;
  operator tensor::Tensor&() const;
  template<typename TValue,
           typename = typename std::enable_if<
           std::is_base_of<Value, TValue>::value>::type>
  explicit operator TValue () const {
    return ir::Downcast<TValue>(*this);
  }
  MNM_DEF_NODE_REF_METHODS(Value, ir::NodeRef, ValueNode);
};

/* TensorValue */
class TensorValueNode final : public ValueNode {
 public:
  mutable tensor::Tensor tensor;
  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("_tensor", &tensor);
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
    v->Visit("_fields", &fields);
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
    v->Visit("_env", &env);
    v->Visit("_func", &func);
  }
  static constexpr const char* _type_key = "mnm.value.ClosureValue";
  MNM_DEF_NODE_TYPE_INFO(ClosureValueNode, ValueNode);
};

class ClosureValue final : public Value {
 public:
  static ClosureValue make(ir::Map<ir::Var, Value> env, ir::Function func);
  MNM_DEF_NODE_REF_METHODS(ClosureValue, Value, ClosureValueNode);
};

/* RefValue */
class RefValueNode final : public ValueNode {
 public:
  mutable Value value;
  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("_value", &value);
  }
  static constexpr const char* _type_key = "mnm.value.RefValue";
  MNM_DEF_NODE_TYPE_INFO(RefValueNode, ValueNode);
};

class RefValue final : public Value {
 public:
  static RefValue make(Value value);
  MNM_DEF_NODE_REF_METHODS(RefValue, Value, RefValueNode);
};

/* OpValue */
class OpValueNode final : public ValueNode {
 public:
  ir::Op op;
  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("_op", &op);
  }
  static constexpr const char* _type_key = "mnm.value.OpValue";
  MNM_DEF_NODE_TYPE_INFO(OpValueNode, ValueNode);
};

class OpValue final : public Value {
 public:
  static OpValue make(ir::Op op);
  MNM_DEF_NODE_REF_METHODS(OpValue, Value, OpValueNode);
};

/* ConstructorValue */
class ConstructorValueNode;
class ConstructorValue;

/* OpaqueValue */
class OpaqueValueNode : public ValueNode {
 public:
  mutable Value data{nullptr};
  static constexpr const char* _type_key = "mnm.value.OpaqueValue";
  MNM_DEF_NODE_TYPE_INFO(OpaqueValueNode, ValueNode);
};

class OpaqueValue : public Value {
 public:
  MNM_DEF_NODE_REF_METHODS(OpaqueValue, Value, OpaqueValueNode);
};

}  // namespace value
}  // namespace mnm

// Scalar values

namespace mnm {
namespace value {

class IntValue;
class FloatValue;
class BoolValue;

/* ScalarValue */
class ScalarValueNode : public ValueNode {
 public:
  static constexpr const char* _type_key = "mnm.value.ScalarValue";
  MNM_DEF_BASE_NODE_INFO(ScalarValueNode, ValueNode);
};

class ScalarValue : public Value {
 public:
  static IntValue make(int data);
  static IntValue make(int64_t data);
  static FloatValue make(double data);
  static BoolValue make(bool data);
  MNM_DEF_NODE_REF_METHODS(ScalarValue, Value, ScalarValueNode);
};

/* IntValue */
class IntValueNode : public ScalarValueNode {
 public:
  int64_t data;
  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("data", &data);
  }
  static constexpr const char* _type_key = "mnm.value.IntValue";
  MNM_DEF_NODE_TYPE_INFO(IntValueNode, ScalarValueNode);
};

class IntValue : public ScalarValue {
 public:
  static IntValue make(int64_t data);
  MNM_DEF_NODE_REF_METHODS(IntValue, ScalarValue, IntValueNode);
};

/* FloatValue */
class FloatValueNode : public ScalarValueNode {
 public:
  double data;
  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("data", &data);
  }
  static constexpr const char* _type_key = "mnm.value.FloatValue";
  MNM_DEF_NODE_TYPE_INFO(FloatValueNode, ScalarValueNode);
};

class FloatValue : public ScalarValue {
 public:
  static FloatValue make(double data);
  MNM_DEF_NODE_REF_METHODS(FloatValue, ScalarValue, FloatValueNode);
};

/* BoolValue */
class BoolValueNode : public ScalarValueNode {
 public:
  bool data;
  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("data", &data);
  }
  static constexpr const char* _type_key = "mnm.value.BoolValue";
  MNM_DEF_NODE_TYPE_INFO(BoolValueNode, ScalarValueNode);
};

class BoolValue : public ScalarValue {
 public:
  static BoolValue make(bool data);
  MNM_DEF_NODE_REF_METHODS(BoolValue, ScalarValue, BoolValueNode);
};

/* StringValue */
class StringValueNode : public ValueNode {
 public:
  std::string data;
  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("data", &data);
  }
  static constexpr const char* _type_key = "mnm.value.StringValue";
  MNM_DEF_NODE_TYPE_INFO(StringValueNode, ValueNode);
};

class StringValue : public Value {
 public:
  static StringValue make(const std::string &data);
  MNM_DEF_NODE_REF_METHODS(StringValue, Value, StringValueNode);
};

}  // namespace value
}  // namespace mnm

namespace mnm {
namespace value {

class BoundExprNode : public ir::Node {
 public:
  ir::Expr expr;
  Value value;
  mutable executor::Executor* executor{nullptr};

  ~BoundExprNode();

  void BindExecutor(executor::Executor* executor);

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("_expr", &expr);
    v->Visit("_value", &value);
  }

  static constexpr const char* _type_key = "mnm.value.BoundExpr";
  MNM_DEF_NODE_TYPE_INFO(BoundExprNode, ir::Node);
};

class BoundExpr : public ir::NodeRef {
 public:
  MNM_DEF_NODE_REF_METHODS(BoundExpr, ir::NodeRef, BoundExprNode);
  static BoundExpr make(ir::Expr expr, Value value);
};

ir::Type GetType(const Value &value);

}  // namespace value
}  // namespace mnm
