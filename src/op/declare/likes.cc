/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/declare/unary.cc
 * \brief Declaration of unary operators
 */
#include "raf/op.h"
#include "raf/tensor.h"
#include "../schema/likes.h"
#include <numeric>
#include <algorithm>
namespace raf {
namespace op {
namespace declare {

using namespace raf::op::schema;
using namespace raf::value;

void Sum(const CallValues& call) {
  const auto* args = call->args.as<SumArgs>();
  CHECK(args != nullptr);
  DLTensor* x = args->x;
  auto ndim = x->ndim;
  auto exclude = args->exclude;
  // Sort the axis
  std::vector<int64_t> axis = args->axis;
  std::vector<int64_t> keep = args->keepdims;
  for (auto& x : axis) {
    x = (x % ndim + ndim) % ndim;
  }
  if (exclude && (keep.size() != 1)) {
    LOG(FATAL) << "invalid combination of argument in sum op";
  }
  if (exclude && axis.empty()) {
    LOG(FATAL) << "invalid combination of argument in sum op";
  }
  if (exclude) {
    std::vector<int64_t> axis_exclude;
    for (int i = 0; i < ndim; i++) {
      auto it = std::find(axis.begin(), axis.end(), i);
      if (it == axis.end()) {
        axis_exclude.push_back(i);
      }
    }
    axis = axis_exclude;
  }
  if (axis.empty() && !keep.empty()) {
    axis.resize(ndim);
    std::iota(axis.begin(), axis.end(), 0);
  }
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
  call->device = x->device;
  call->out = TensorValue::Assemble(x->device, x->dtype, shape);
}

RAF_OP_DECLARE("raf.op.sum", Sum);

void SumDx(const CallValues& call) {
  // the shape of the output of reduce_dx op is same as input x
  const auto* args = call->args.as<SumDxArgs>();
  CHECK(args != nullptr);
  DLTensor* x = args->x;
  std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
  call->device = x->device;
  call->out = TensorValue::Assemble(/*dev=*/x->device,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/shape);
}
RAF_OP_DECLARE("raf.op.sum_dx", SumDx);
}  // namespace declare
}  // namespace op
}  // namespace raf
