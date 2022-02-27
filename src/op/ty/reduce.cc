/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/ty/reduce.cc
 * \brief Typing of reduction operators
 */
#include <iostream>
#include <algorithm>
#include <numeric>
#include <tvm/relay/type.h>
#include "raf/op_utils.h"
#include "raf/type.h"
#include "../schema/likes.h"
#include "../schema/reduce.h"
#include "./utils.h"

namespace raf {
namespace op {

using namespace raf::ir;
using namespace raf::value;
using schema::ProdDxArgs;
using schema::ReduceArgs;
using schema::SumArgs;
using schema::SumDxArgs;

Type SumInfer(const CallValues& value) {
  const auto* args = value->args.as<SumArgs>();
  CHECK(args != nullptr);
  TensorType x = Downcast<TensorType>(GetType(args->x));
  auto ndim = x->shape.size();
  auto exclude = args->exclude;

  // Sort the axis
  std::vector<int64_t> axis = args->axis;
  std::vector<int64_t> keep = args->keepdims;
  for (auto& x : axis) {
    x = x < 0 ? x + (int64_t)ndim : x;
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
    if (axis[i] < 0) {
      axis[i] += x->shape.size();
    }
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

RAF_OP_TYPE("raf.op.sum", "Sum", SumInfer);

Type SumDxInfer(const CallValues& value) {
  const auto* args = value->args.as<SumDxArgs>();
  CHECK(args != nullptr);
  return GetType(args->x);
}
RAF_OP_TYPE("raf.op.sum_dx", "SumDx", SumDxInfer);

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
  bool exclude = args->exclude;
  if (exclude) {
    std::vector<int64_t> axis_exclude;
    for (int64_t i = 0; i < ndim; i++) {
      if (std::find(axis.begin(), axis.end(), i) == axis.end()) {
        axis_exclude.push_back(i);
      }
    }
    axis = axis_exclude;
  }
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
  return TensorType(shape, tvm::runtime::DataType::Int(32));
}

Type ReduceOutSameDType(const CallValues& value) {
  const auto* args = value->args.as<ReduceArgs>();
  CHECK(args != nullptr);
  TensorType x = Downcast<TensorType>(GetType(args->x));
  Array<PrimExpr> shape = GenerateReduceShape(args, x);
  return TensorType(shape, x->dtype);
}

Type ProdDxDType(const CallValues& value) {
  const auto* args = value->args.as<ProdDxArgs>();
  CHECK(args != nullptr);
  TensorType x = Downcast<TensorType>(GetType(args->x));
  return x;
}

Type MeanDxInfer(const CallValues& value) {
  const auto* args = value->args.as<schema::MeanDxArgs>();
  CHECK(args != nullptr);
  TensorType dy = Downcast<TensorType>(GetType(args->dy));
  Array<tvm::PrimExpr> oshape = GetShapeExprFromValue(args->shape);
  return TensorType(oshape, dy->dtype);
}

RAF_OP_TYPE("raf.op.argmax", "Argmax", ReduceOutIntDType);
RAF_OP_TYPE("raf.op.argmin", "Argmin", ReduceOutIntDType);
RAF_OP_TYPE("raf.op.max", "Max", ReduceOutSameDType);
RAF_OP_TYPE("raf.op.min", "Min", ReduceOutSameDType);
RAF_OP_TYPE("raf.op.all", "All", ReduceOutSameDType);
RAF_OP_TYPE("raf.op.any", "Any", ReduceOutSameDType);
RAF_OP_TYPE("raf.op.prod", "Prod", ReduceOutSameDType);
RAF_OP_TYPE("raf.op.mean", "Mean", ReduceOutSameDType);
RAF_OP_TYPE("raf.op.prod_dx", "ProdDx", ProdDxDType);
RAF_OP_TYPE("raf.op.mean_dx", "MeanDx", MeanDxInfer);

Type L2NormDType(const CallValues& value) {
  const auto* args = value->args.as<schema::L2NormArgs>();
  CHECK(args != nullptr);
  TensorType x = Downcast<TensorType>(GetType(args->x));
  Array<tvm::PrimExpr> oshape;
  return TensorType(oshape, x->dtype);
}

RAF_OP_TYPE("raf.op.l2norm", "L2norm", L2NormDType);
}  // namespace op
}  // namespace raf
