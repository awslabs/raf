/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/declare/unary.cc
 * \brief Declaration of unary operators
 */
#include "mnm/op.h"
#include "mnm/tensor.h"
#include "../schema/likes.h"

namespace mnm {
namespace op {
namespace declare {

using namespace mnm::op::schema;
using namespace mnm::value;

void Sum(const CallValues& call) {
  const auto* args = call->args.as<SumArgs>();
  CHECK(args != nullptr);
  DLTensor* x = args->x;
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
  std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
  for (int i = 0, n = axis_info.size(); i < n; ++i) {
    int64_t& cur_axis = axis_info[i].first;
    cur_axis = (cur_axis + x->ndim) % x->ndim;
    CHECK(cur_axis >= 0 && cur_axis < x->ndim);
    shape[cur_axis] = 1;
    if (i) {
      CHECK_NE(axis_info[i].first, axis_info[i - 1].first) << "Cannot collapse repeated axis!";
    }
  }
  // Squeeze the unkept dims
  for (int i = 0, n = axis_info.size(), j = 0; i < n; ++i) {
    if (!axis_info[i].second) {
      shape.erase(shape.begin() + axis[i] - j);
      ++j;
    }
  }
  call->ctx = x->ctx;
  call->out = TensorValue::Assemble(x->ctx, x->dtype, shape);
}

MNM_OP_DECLARE("mnm.op.sum", Sum).set_attr<TOpPattern>("TOpPattern", kCommReduce);

}  // namespace declare
}  // namespace op
}  // namespace mnm
