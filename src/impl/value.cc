#include <tvm/runtime/ndarray.h>

#include <mnm/executor.h>
#include <mnm/ir.h>
#include <mnm/registry.h>
#include <mnm/tensor.h>
#include <mnm/value.h>

#include "../common/shape_utils.h"

namespace mnm {
namespace value {

using common::shape_utils::GetShape;
using common::shape_utils::MakeShape;
using executor::Executor;
using ir::Array;
using ir::Downcast;
using ir::Expr;
using ir::Function;
using ir::Integer;
using ir::make_node;
using ir::Map;
using ir::NodePtr;
using ir::NodeRef;
using ir::Op;
using ir::TensorType;
using ir::TupleType;
using ir::Type;
using ir::Var;
using tensor::Tensor;

/*** Constructors ***/
TensorValue TensorValue::make(tensor::Tensor tensor) {
  NodePtr<TensorValueNode> n = make_node<TensorValueNode>();
  n->tensor = std::move(tensor);
  return TensorValue(n);
}

TupleValue TupleValue::make(Array<Value> fields) {
  NodePtr<TupleValueNode> n = make_node<TupleValueNode>();
  n->fields = std::move(fields);
  return TupleValue(n);
}

ClosureValue ClosureValue::make(Map<Var, Value> env, Function func) {
  NodePtr<ClosureValueNode> n = make_node<ClosureValueNode>();
  n->env = std::move(env);
  n->func = std::move(func);
  return ClosureValue(n);
}

RefValue RefValue::make(Value value) {
  NodePtr<RefValueNode> n = make_node<RefValueNode>();
  n->value = std::move(value);
  return RefValue(n);
}

OpValue OpValue::make(Op op) {
  NodePtr<OpValueNode> n = make_node<OpValueNode>();
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
  NodePtr<IntValueNode> n = make_node<IntValueNode>();
  n->data = data;
  return IntValue(n);
}

FloatValue FloatValue::make(double data) {
  NodePtr<FloatValueNode> n = make_node<FloatValueNode>();
  n->data = data;
  return FloatValue(n);
}

BoolValue BoolValue::make(bool data) {
  NodePtr<BoolValueNode> n = make_node<BoolValueNode>();
  n->data = data;
  return BoolValue(n);
}

StringValue StringValue::make(const std::string& data) {
  NodePtr<StringValueNode> n = make_node<StringValueNode>();
  n->data = data;
  return StringValue(n);
}

BoundExpr BoundExpr::make(Expr expr, Value value) {
  NodePtr<BoundExprNode> n = make_node<BoundExprNode>();
  n->expr = std::move(expr);
  n->value = std::move(value);
  return BoundExpr(n);
}

/*** BoundExpr ***/
BoundExprNode::~BoundExprNode() {
  if (executor != nullptr) {
    executor->OnDestruct(this);
  }
}

void BoundExprNode::BindExecutor(Executor* executor) {
  CHECK(this->executor == nullptr);
  this->executor = executor;
  executor->OnBind(this);
}

/*** GetType ***/
Type GetType(const Value& value) {
  if (const auto* tv = value.as<TensorValueNode>()) {
    const DLTensor& dlt = *tv->tensor.operator->();
    auto shape = GetShape<tvm::Integer>(dlt);
    return ir::TensorTypeNode::make({shape.begin(), shape.end()}, tvm::TVMType2Type(dlt.dtype));
  } else if (const auto* tv = value.as<TupleValueNode>()) {
    Array<Type> tuple_type;
    for (const Value& sub_value : tv->fields) {
      tuple_type.push_back(GetType(sub_value));
    }
    return ir::TupleTypeNode::make(tuple_type);
  }
  LOG(FATAL) << "NotImplementedError: " << value->type_key();
  throw;
}

/*** Value ***/
Value::operator const DLTensor*() const {
  if (const auto* tensor_value = this->as<TensorValueNode>()) {
    const DLTensor* dl_tensor_ref = tensor_value->tensor.operator->();
    return dl_tensor_ref;
  }
  LOG(FATAL) << "InternalError: cannot convert to TensorValue";
  throw;
}

Value::operator const tensor::Tensor&() const {
  if (const auto* tensor_value = this->as<TensorValueNode>()) {
    return tensor_value->tensor;
  }
  LOG(FATAL) << "InternalError: cannot convert to TensorValue";
  throw;
}

#define MNM_SWITCH_SCALAR(var, value, body)                      \
  do                                                             \
    if (const auto* var = (value).as<IntValueNode>()) {          \
      body;                                                      \
    } else if (const auto* var = (value).as<FloatValueNode>()) { \
      body;                                                      \
    } else if (const auto* var = (value).as<BoolValueNode>()) {  \
      body;                                                      \
    }                                                            \
  while (0);

Value::operator int() const {
  MNM_SWITCH_SCALAR(value, *this, { return value->data; });
  LOG(FATAL) << "InternalError: cannot be converted to int";
  throw;
}

Value::operator int64_t() const {
  MNM_SWITCH_SCALAR(value, *this, { return value->data; });
  LOG(FATAL) << "InternalError: cannot be converted to int64_t";
  throw;
}

Value::operator float() const {
  MNM_SWITCH_SCALAR(value, *this, { return value->data; });
  LOG(FATAL) << "InternalError: cannot be converted to float";
  throw;
}

Value::operator double() const {
  MNM_SWITCH_SCALAR(value, *this, { return value->data; });
  LOG(FATAL) << "InternalError: cannot be converted to double";
  throw;
}

Value::operator bool() const {
  MNM_SWITCH_SCALAR(value, *this, { return value->data; });
  LOG(FATAL) << "InternalError: cannot be converted to bool";
  throw;
}

Value::operator std::string() const {
  if (const auto* value = this->as<StringValueNode>()) {
    return value->data;
  }
  LOG(FATAL) << "InternalError: cannot be converted to std::string";
  throw;
}

/*** TensorValue ***/
TensorValue TensorValue::Assemble(const Context& ctx, const DType& dtype,
                                  const std::vector<int64_t>& shape,
                                  const std::vector<int64_t>& strides, void* const data) {
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

NodeRef DeTuple(Value value) {
  if (value->is_type<TensorValueNode>()) {
    return std::move(value);
  }
  if (const auto* tuple = value.as<TupleValueNode>()) {
    Array<NodeRef> result;
    for (Value sub_value : tuple->fields) {
      if (sub_value->op_env == nullptr) {
        sub_value->op_env = tuple->op_env;
      }
      result.push_back(DeTuple(sub_value));
    }
    return std::move(result);
  }
  LOG(FATAL) << "ValueError: cannot de-tuple " << value->type_key();
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
MNM_REGISTER_GLOBAL("mnm.value._make.BoundExpr").set_body_typed(BoundExpr::make);

}  // namespace value
}  // namespace mnm
