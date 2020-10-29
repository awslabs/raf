/*!
 * Copyright (c) 2020 by Contributors
 * \file src/op/ty/reduce.cc
 * \brief Typing of reduction operators
 */
#include <tvm/relay/type.h>
#include <numeric>
#include "mnm/type.h"
#include "../schema/likes.h"
#include "../schema/reduce.h"
#include "./utils.h"

namespace mnm {
namespace op {
namespace type {

using namespace mnm::value;
using schema::ReduceArgs;
using schema::SumArgs;
using tvm::relay::Type;
using namespace tvm;
using namespace tvm::relay;

Type SumInfer(const CallValues& value) {
  using namespace tvm;
  using namespace tvm::relay;
  const auto* args = value->args.as<SumArgs>();
  CHECK(args != nullptr);
  TensorType x = Downcast<TensorType>(GetType(args->x));
  // Sort the axis
  std::vector<int64_t> axis = args->axis;
  std::vector<int64_t> keep = args->keepdims;
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

Array<PrimExpr> GenerateReduceShape(const ReduceArgs* args, tvm::TensorType x) {
  Array<PrimExpr> shape;
  auto ndim = x->shape.size();
  std::vector<int64_t> axis;
  if (args->axis.empty()) {
    axis.resize(ndim);
    std::iota(axis.begin(), axis.end(), 0);
  } else {
    axis = args->axis;
  }
  for (size_t i = 0; i < axis.size(); i++) {
    if (axis[i] < 0) {
      axis[i] += ndim;
    }
  }
  std::sort(axis.begin(), axis.end());
  axis.resize(std::unique(axis.begin(), axis.end()) - axis.begin());
  bool keepdims = args->keepdims;
  if (keepdims) {
    for (size_t i = 0; i < ndim; i++) {
      if (std::find(axis.begin(), axis.end(), i) != axis.end()) {
        shape.push_back(IntImm(x->shape[i].dtype(), 1));
      } else {
        shape.push_back(x->shape[i]);
      }
    }
  } else {
    for (size_t i = 0; i < ndim; i++) {
      if (std::find(axis.begin(), axis.end(), i) == axis.end()) {
        shape.push_back(x->shape[i]);
      }
    }
  }
  return shape;
}

Type ReduceOutIntDType(const CallValues& value) {
  const auto* args = value->args.as<ReduceArgs>();
  CHECK(args != nullptr);
  TensorType x = Downcast<TensorType>(GetType(args->x));
  Array<PrimExpr> shape = GenerateReduceShape(args, x);
  return TensorType(shape, tvm::runtime::DataType::Int(64));
}

Type ReduceOutSameDType(const CallValues& value) {
  const auto* args = value->args.as<ReduceArgs>();
  CHECK(args != nullptr);
  TensorType x = Downcast<TensorType>(GetType(args->x));
  Array<PrimExpr> shape = GenerateReduceShape(args, x);
  return TensorType(shape, x->dtype);
}

MNM_OP_TYPE("mnm.op.argmax", "Argmax", ReduceOutIntDType);
MNM_OP_TYPE("mnm.op.argmin", "Argmin", ReduceOutIntDType);
MNM_OP_TYPE("mnm.op.max", "Max", ReduceOutSameDType);
MNM_OP_TYPE("mnm.op.min", "Min", ReduceOutSameDType);
MNM_OP_TYPE("mnm.op.all", "All", ReduceOutSameDType);
MNM_OP_TYPE("mnm.op.any", "Any", ReduceOutSameDType);
MNM_OP_TYPE("mnm.op.mean", "Mean", ReduceOutSameDType);
MNM_OP_TYPE("mnm.op.mean_dx", "MeanDx", ReduceOutSameDType);

}  // namespace type
}  // namespace op
}  // namespace mnm
