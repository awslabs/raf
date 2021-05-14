/*!
 * Copyright (c) 2019 by Contributors
 * \file ./src/op/from_relay/from_relay_utils.cc
 * \brief Utility methods for Relay to Meta op conversion.
 */
#include "./from_relay_utils.h"

using namespace mnm::value;
using namespace tvm;
using namespace ::tvm::relay;

namespace mnm {
namespace op {
namespace from_relay {

std::vector<int64_t> ArrayToInt(const Array<IndexExpr>& arr) {
  std::vector<int64_t> ret;
  for (const auto i : arr) {
    auto node = i.as<IntImmNode>();
    CHECK(node != nullptr) << "Array elemment " << i << " is not IntImmNode";
    int64_t val = node->value;
    ret.push_back(val);
  }
  return std::move(ret);
}

std::vector<int64_t> ArrayToInt(const Array<Integer>& arr) {
  std::vector<int64_t> ret;
  for (const auto i : arr) {
    auto node = i.as<IntImmNode>();
    CHECK(node != nullptr) << "Array elemment " << i << " is not IntImmNode";
    int64_t val = node->value;
    ret.push_back(val);
  }
  return std::move(ret);
}

std::vector<int64_t> ArrayToInt(const ArrayNode& arr) {
  std::vector<int64_t> ret;
  for (const auto i : arr) {
    auto node = i.as<IntImmNode>();
    CHECK(node != nullptr) << "Array elemment " << i << " is not IntImmNode";
    int64_t val = node->value;
    ret.push_back(val);
  }
  return std::move(ret);
}

TupleValue ArrayToIntTuple(const Array<IndexExpr>& arr) {
  Array<Value> ret;
  for (auto val : ArrayToInt(arr)) {
    ret.push_back(ScalarValue::make(val));
  }
  return TupleValue::make(std::move(ret));
}

TupleValue ArrayToIntTuple(const Array<Integer>& arr) {
  Array<Value> ret;
  for (auto val : ArrayToInt(arr)) {
    ret.push_back(ScalarValue::make(val));
  }
  return TupleValue::make(std::move(ret));
}

TupleValue ArrayToIntTuple(const std::vector<int64_t>& arr) {
  Array<Value> ret;
  for (auto val : arr) {
    ret.push_back(ScalarValue::make(val));
  }
  return TupleValue::make(std::move(ret));
}

TupleValue ArrayToIntTuple(const ArrayNode& arr) {
  Array<Value> ret;
  for (auto val : ArrayToInt(arr)) {
    ret.push_back(ScalarValue::make(val));
  }
  return TupleValue::make(std::move(ret));
}

Var GetMayShare(const Expr& var) {
  const auto* vn = var.as<ExtendedVarNode>();
  CHECK(vn);
  while (vn->may_share.defined()) {
    vn = vn->may_share.as<ExtendedVarNode>();
    CHECK(vn);
  }
  return GetRef<Var>(vn);
}

}  // namespace from_relay
}  // namespace op
}  // namespace mnm
