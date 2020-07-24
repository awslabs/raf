/*!
 * Copyright (c) 2020 by Contributors
 * \file src/op/ty/reduce.cc
 * \brief Typing of reduction operators
 */
#include <tvm/relay/type.h>
#include "mnm/type.h"
#include "../schema/likes.h"
#include "./utils.h"

namespace mnm {
namespace op {
namespace type {

using namespace mnm::value;
using tvm::relay::Type;
using schema::SumArgs;

Type SumInfer(const CallValues& value) {
  using namespace tvm;
  using namespace tvm::relay;
  const auto* args = value->args.as<SumArgs>();
  CHECK(args != nullptr);
  TensorType x = Downcast<TensorType>(GetType(args->x));
  // Sort the axis
  std::vector<int64_t> axis = args->axis;
  std::vector<int64_t> keep = args->keep;
  if (keep.size() == 1) {
    keep = std::vector<int64_t>(axis.size(), keep[0]);
  }
  CHECK_EQ(axis.size(), keep.size());
  std::vector<std::pair<int64_t, int64_t>> axis_info;
  for (int i = 0, n = axis.size(); i < n; ++i) {
    axis_info.emplace_back(axis[i], keep[i]);
  }
  std::sort(axis_info.begin(), axis_info.end());
  // Figure out the shape
  Array<PrimExpr> shape;
  int m = static_cast<int>(axis_info.size());
  int n = static_cast<int>(x->shape.size());
  for (int i = 1; i < m; ++i) {
    CHECK_NE(axis_info[i].first, axis_info[i - 1].first) << "Cannot collapse repeated axis!";
  }
  for (int i = 0, j = 0; i < n; ++i) {
    if (j < m && axis_info[j].first == i) {
      // keep the collapsed axis
      if (axis_info[j].second) {
        shape.push_back(IntImm(x->shape[i].dtype(), 1));
      }
      ++j;
    } else {
      shape.push_back(x->shape[i]);
    }
  }
  return TensorType(shape, x->dtype);
}

MNM_OP_TYPE("mnm.op.sum", "Sum", SumInfer);


}  // namespace type
}  // namespace op
}  // namespace mnm
