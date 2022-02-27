/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file ./src/op/dialect/tvm_dialect/binary.cc
 * \brief Binary operators bridged from TVM.
 */
#include <vector>
#include <numeric>
#include "./tvm_attrs.h"
#include "./tvm_utils.h"
#include "../../schema/reduce.h"
#include "../../../common/shape_utils.h"
#include "../../schema/likes.h"

namespace raf {
namespace op {
namespace tvm_dialect {

using namespace raf::ir;
using namespace raf::op::schema;
using common::shape_utils::GetNumel;

std::vector<Value> ReduceSchema2Args(const ReduceArgs* args) {
  return {args->x};
}

std::vector<std::string> ReduceSchemaArgNames(const op::CallValues& call) {
  return {"x"};
}

Attrs ReduceSchema2Attrs(const ReduceArgs* args) {
  auto attrs = make_object<ReduceAttrs>();
  // expand the empty axis
  DLTensor* x = args->x;
  auto ndim = x->ndim;
  std::vector<int64_t> axis;
  if (args->axis.empty()) {
    axis.resize(ndim);
    std::iota(axis.begin(), axis.end(), 0);
  } else {
    axis = args->axis;
  }
  for (int i = 0, n = axis.size(); i < n; ++i) {
    attrs->axis.push_back(axis[i]);
  }
  attrs->keepdims = args->keepdims;
  attrs->exclude = args->exclude;
  return Attrs(attrs);
}

HashKey ReduceHasher(const std::vector<Type>& param_types, const Type& ret_type,
                     const ReduceArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, ret_type, nullptr);
  for (int i = 0, n = args->axis.size(); i < n; ++i) {
    key << args->axis[i];
  }
  key << args->keepdims;
  key << args->exclude;
  return key;
}

#define RAF_TVM_REDUCE(OP, FUNC)                                                             \
  RAF_TVM(OP, FUNC, ReduceArgs, ReduceSchema2Args, ReduceSchemaArgNames, ReduceSchema2Attrs, \
          ReduceHasher, kCommReduce)

RAF_TVM_REDUCE(max, Max);
RAF_TVM_REDUCE(min, Min);
RAF_TVM_REDUCE(all, All);
RAF_TVM_REDUCE(any, Any);
RAF_TVM_REDUCE(mean, Mean);
RAF_TVM_REDUCE(prod, Prod);

Attrs ReduceSchema2ArgReduceAttrs(const ReduceArgs* args) {
  auto attrs = make_object<ArgReduceAttrs>();
  // expand the empty axis
  DLTensor* x = args->x;
  auto ndim = x->ndim;
  std::vector<int64_t> axis;
  if (args->axis.empty()) {
    axis.resize(ndim);
    std::iota(axis.begin(), axis.end(), 0);
  } else {
    axis = args->axis;
  }
  for (int i = 0, n = axis.size(); i < n; ++i) {
    attrs->axis.push_back(axis[i]);
  }
  attrs->keepdims = args->keepdims;
  attrs->exclude = args->exclude;
  attrs->select_last_index = false;
  return Attrs(attrs);
}

RAF_TVM(argmax, Argmax, ReduceArgs, ReduceSchema2Args, ReduceSchemaArgNames,
        ReduceSchema2ArgReduceAttrs, ReduceHasher, kCommReduce);
RAF_TVM(argmin, Argmin, ReduceArgs, ReduceSchema2Args, ReduceSchemaArgNames,
        ReduceSchema2ArgReduceAttrs, ReduceHasher, kCommReduce)

std::vector<Value> ProdDxSchema2Args(const ProdDxArgs* args) {
  return {args->x, args->dy};
}

std::vector<std::string> ProdDxSchemaArgNames(const op::CallValues& call) {
  return {"x", "dy"};
}

Attrs ProdDxSchema2Attrs(const ProdDxArgs* args) {
  auto attrs = make_object<ReduceAttrs>();
  // expand the empty axis
  DLTensor* x = args->x;
  auto ndim = x->ndim;
  std::vector<int64_t> axis;
  if (args->axis.empty()) {
    axis.resize(ndim);
    std::iota(axis.begin(), axis.end(), 0);
  } else {
    axis = args->axis;
  }
  for (int i = 0, n = axis.size(); i < n; ++i) {
    attrs->axis.push_back(axis[i]);
  }
  attrs->keepdims = args->keepdims;
  attrs->exclude = args->exclude;
  return Attrs(attrs);
}

HashKey ProdDxHasher(const std::vector<Type>& param_types, const Type& ret_type,
                     const ProdDxArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, ret_type, nullptr);
  for (int i = 0, n = args->axis.size(); i < n; ++i) {
    key << args->axis[i];
  }
  key << args->keepdims;
  key << args->exclude;
  return key;
}

RAF_TVM(prod_dx, ProdDx, ProdDxArgs, ProdDxSchema2Args, ProdDxSchemaArgNames, ProdDxSchema2Attrs,
        ProdDxHasher, kBroadcast);

std::vector<Value> SumSchema2Args(const SumArgs* args) {
  return {args->x};
}

std::vector<std::string> SumSchemaArgNames(const op::CallValues& call) {
  return {"x"};
}

Attrs SumSchema2Attrs(const SumArgs* args) {
  auto attrs = make_object<SumAttrs>();
  DLTensor* x = args->x;
  CHECK(x);
  for (int i = 0, n = args->axis.size(); i < n; ++i) {
    attrs->axis.push_back((args->axis[i] % x->ndim + x->ndim) % x->ndim);
  }
  for (int i = 0, n = args->keepdims.size(); i < n; ++i) {
    attrs->keepdims.push_back(args->keepdims[i]);
  }
  attrs->exclude = args->exclude;
  return Attrs(attrs);
}

HashKey SumHasher(const std::vector<Type>& param_types, const Type& ret_type, const SumArgs* args) {
  HashKey key = GenericHasher<std::nullptr_t>(param_types, ret_type, nullptr);
  for (int i = 0, n = args->axis.size(); i < n; ++i) {
    key << args->axis[i];
    key << args->keepdims[i];
  }
  key << args->exclude;
  return key;
}

std::vector<Value> SumDxSchema2Args(const SumDxArgs* args) {
  return {args->x, args->dy};
}

std::vector<std::string> SumDxSchemaArgNames(const op::CallValues& call) {
  return {"x", "dy"};
}

Attrs SumDxSchema2Attrs(const SumDxArgs* args) {
  auto attrs = make_object<SumAttrs>();
  // expand the empty axis
  DLTensor* x = args->x;
  std::vector<int64_t> axis;
  for (int i = 0, n = args->axis.size(); i < n; ++i) {
    attrs->axis.push_back(args->axis[i]);
  }
  for (int i = 0, n = args->keepdims.size(); i < n; ++i) {
    attrs->keepdims.push_back(args->keepdims[i]);
  }
  attrs->exclude = args->exclude;
  return Attrs(attrs);
}

HashKey SumDxHasher(const std::vector<Type>& param_types, const Type& ret_type,
                    const SumDxArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, ret_type, nullptr);
  for (int i = 0, n = args->axis.size(); i < n; ++i) {
    key << args->axis[i];
    key << args->keepdims[i];
  }
  key << args->exclude;
  return key;
}

RAF_TVM(sum, Sum, SumArgs, SumSchema2Args, SumSchemaArgNames, SumSchema2Attrs, SumHasher,
        kCommReduce);

RAF_TVM(sum_dx, SumDx, SumDxArgs, SumDxSchema2Args, SumDxSchemaArgNames, SumDxSchema2Attrs,
        SumDxHasher, kBroadcast);

std::vector<Value> MeanDxSchema2Args(const MeanDxArgs* args) {
  return {args->dy};
}

std::vector<std::string> MeanDxSchemaArgNames(const op::CallValues& call) {
  return {"dy"};
}

Attrs MeanDxSchema2Attrs(const MeanDxArgs* args) {
  auto attrs = make_object<MeanDxAttrs>();
  std::vector<int64_t> shape = GetShapeVecFromValue(args->shape);
  auto ndim = shape.size();
  for (int64_t s : shape) {
    attrs->shape.push_back(s);
  }
  std::vector<int64_t> axis;
  if (args->axis.empty()) {
    axis.resize(ndim);
    std::iota(axis.begin(), axis.end(), 0);
  } else {
    axis = args->axis;
  }
  for (int i = 0, n = axis.size(); i < n; ++i) {
    attrs->axis.push_back(axis[i]);
  }
  attrs->keepdims = args->keepdims;
  attrs->exclude = args->exclude;
  return Attrs(attrs);
}

HashKey MeanDxHasher(const std::vector<Type>& param_types, const Type& ret_type,
                     const MeanDxArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, ret_type, nullptr);
  key << args->axis;
  key << args->keepdims;
  key << args->exclude;
  return key;
}

RAF_TVM(mean_dx, MeanDx, MeanDxArgs, MeanDxSchema2Args, MeanDxSchemaArgNames, MeanDxSchema2Attrs,
        MeanDxHasher, kBroadcast);

std::vector<Value> L2NormSchema2Args(const L2NormArgs* args) {
  return {args->x};
}

std::vector<std::string> L2NormSchemaArgNames(const op::CallValues& call) {
  return {"x"};
}

RAF_TVM(l2norm, L2Norm, L2NormArgs, L2NormSchema2Args, L2NormSchemaArgNames, GenericAttrs,
        GenericHasher, kCommReduce);

}  // namespace tvm_dialect
}  // namespace op
}  // namespace raf
