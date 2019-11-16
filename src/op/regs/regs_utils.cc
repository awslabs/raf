/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/regs/regs_utils.cc
 * \brief Helpers for operator registry
 */
#include "mnm/tensor.h"
#include "mnm/value.h"
#include "./regs_utils.h"
#include "../schema/list_args.h"

namespace mnm {
namespace op {

using ir::Array;
using ir::Attrs;
using ir::make_node;
using value::Value;

Attrs MakeListArgs(const Array<Value>& values) {
  auto attrs = make_node<op::schema::ListArgs>();
  attrs->args = values;
  return Attrs(attrs);
}

Array<Value> GetListArgs(const Attrs& attrs) {
  return attrs.as<op::schema::ListArgs>()->args;
}

namespace schema {
namespace {
MNM_REGISTER_NODE_TYPE(ListArgs);
}
}  // namespace schema
}  // namespace op
}  // namespace mnm

namespace mnm {
namespace op {
namespace ffi {

using registry::TVMArgValue;
using namespace mnm::ir;
using namespace mnm::value;

#define MNM_CHECK_SYM(a)                                          \
  if ((a).type_code() == kNodeHandle && (a).IsNodeType<Expr>()) { \
    return (a).operator Expr();                                   \
  }

#define MNM_RET_SYM(v_type, v) return MakeConstant(v_type::make(v));

Expr ToAny(const TVMArgValue& a) {
  MNM_CHECK_SYM(a);
  int type_code = a.type_code();
  if (type_code == kDLInt) {
    MNM_RET_SYM(IntValue, a.operator int64_t());
  }
  if (type_code == kDLFloat) {
    MNM_RET_SYM(FloatValue, a.operator double());
  }
  if (type_code == kStr) {
    MNM_RET_SYM(StringValue, a.operator std::string());
  }
  if (type_code == kNull) {
    return MakeConstant(ir::NullValue<Value>());
  }
  if (type_code == kNodeHandle && a.IsNodeType<Array<Integer>>()) {
    return ffi::ToIntTuple(a);
  }
  LOG(FATAL) << "Not supported type code " << type_code;
  throw;
}

Expr ToTensor(const TVMArgValue& a) {
  using mnm::tensor::Tensor;
  MNM_CHECK_SYM(a);
  MNM_RET_SYM(TensorValue, a.AsNDArray<Tensor>());
}

Expr ToInt(const TVMArgValue& a) {
  MNM_CHECK_SYM(a);
  MNM_RET_SYM(IntValue, a.operator int64_t());
}

Expr ToBool(const TVMArgValue& a) {
  MNM_CHECK_SYM(a);
  MNM_RET_SYM(BoolValue, a.operator bool());
}

Expr ToDouble(const TVMArgValue& a) {
  MNM_CHECK_SYM(a);
  MNM_RET_SYM(FloatValue, a.operator double());
}

Expr ToString(const TVMArgValue& a) {
  MNM_CHECK_SYM(a);
  MNM_RET_SYM(StringValue, a.operator std::string());
}

Expr ToIntTuple(const TVMArgValue& a) {
  MNM_CHECK_SYM(a);
  if (a.type_code() == kDLInt) {
    MNM_RET_SYM(TupleValue, {IntValue::make(a.operator int64_t())});
  }
  Array<Value> result;
  for (const auto& item : a.AsNodeRef<Array<Integer>>()) {
    result.push_back(IntValue::make(item.operator int64_t()));
  }
  MNM_RET_SYM(TupleValue, result);
}

Expr ToOptionalIntTuple(const TVMArgValue& a) {
  MNM_CHECK_SYM(a);
  if (a.type_code() == kNull) {
    MNM_RET_SYM(TupleValue, {});
  }
  return ffi::ToIntTuple(a);
}

}  // namespace ffi
}  // namespace op
}  // namespace mnm

namespace mnm {
namespace op {
namespace args {

using namespace mnm::value;
using ir::Downcast;

#define MNM_SWITCH_SCALAR(var, value, body)                      \
  do {                                                          \
    if (const auto* var = (value).as<IntValueNode>()) {          \
      body;                                                      \
    } else if (const auto* var = (value).as<FloatValueNode>()) { \
      body;                                                      \
    } else if (const auto* var = (value).as<BoolValueNode>()) {  \
      body;                                                      \
    }                                                            \
  } while (0);

Value ToAny(const Value& a) {
  return a;
}

TensorValue ToTensor(const Value& a) {
  return Downcast<TensorValue>(a);
}

int64_t ToInt(const Value& a) {
  MNM_SWITCH_SCALAR(value, a, { return value->data; });
  LOG(FATAL) << "InternalError: cannot be converted to int";
  throw;
}

bool ToBool(const Value& a) {
  MNM_SWITCH_SCALAR(value, a, { return value->data; });
  LOG(FATAL) << "InternalError: cannot be converted to int";
  throw;
}

double ToDouble(const Value& a) {
  MNM_SWITCH_SCALAR(value, a, { return value->data; });
  LOG(FATAL) << "InternalError: cannot be converted to double";
  throw;
}

std::string ToString(const value::Value& a) {
  if (const auto* value = a.as<StringValueNode>()) {
    return value->data;
  }
  LOG(FATAL) << "InternalError: cannot be converted to std::string";
  throw;
}

std::vector<int64_t> ToIntTuple(const value::Value& a) {
  if (const auto* v = a.as<IntValueNode>()) {
    return {v->data};
  }
  if (const auto* v = a.as<TupleValueNode>()) {
    std::vector<int64_t> result;
    for (const auto& item : v->fields) {
      if (const auto* vv = item.as<IntValueNode>()) {
        result.push_back(vv->data);
      } else {
        LOG(FATAL) << "Cannot convert to tuple of integers";
        throw;
      }
    }
    return result;
  }
  LOG(FATAL) << "Cannot convert to tuple of integers";
  throw;
}

std::vector<int64_t> ToOptionalIntTuple(const value::Value& a) {
  return a.defined() ? args::ToIntTuple(a) : std::vector<int64_t>();
}

}  // namespace args
}  // namespace op
}  // namespace mnm
