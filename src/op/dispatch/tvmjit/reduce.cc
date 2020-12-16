/*!
 * Copyright (c) 2019 by Contributors
 * \file ./src/op/dispatch/tvmjit/binary.cc
 * \brief Binary operators bridged from TVM.
 */
#include <array>
#include <numeric>
#include "./tvm_attrs.h"
#include "./tvmjit_utils.h"
#include "../../schema/reduce.h"
#include "../../../common/shape_utils.h"

namespace mnm {
namespace op {
namespace tvmjit {

using namespace mnm::ir;
using namespace mnm::op::schema;
using common::shape_utils::GetNumel;

// use tvm::relay::ReduceAttrs here

std::vector<Value> ReduceSchema2Args(const ReduceArgs* args) {
  return {args->x};
}

std::vector<std::string> ReduceSchemaArgNames() {
  return {"x"};
}

Attrs ReduceSchema2Attrs(const ReduceArgs* args) {
  auto attrs = make_object<tvm_attrs::ReduceAttrs>();
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
  attrs->exclude = false;
  return Attrs(attrs);
}

HashKey ReduceHasher(const std::vector<Type>& param_types, const Type& ret_type,
                     const ReduceArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, ret_type, nullptr);
  for (int i = 0, n = args->axis.size(); i < n; ++i) {
    key << args->axis[i];
  }
  key << args->keepdims;
  return key;
}

MNM_TVMJIT(Argmax, "mnm.op.argmax", ReduceArgs, ReduceSchema2Args, ReduceSchemaArgNames,
           ReduceSchema2Attrs, ReduceHasher);
MNM_TVMJIT(Argmin, "mnm.op.argmin", ReduceArgs, ReduceSchema2Args, ReduceSchemaArgNames,
           ReduceSchema2Attrs, ReduceHasher);
MNM_TVMJIT(Max, "mnm.op.max", ReduceArgs, ReduceSchema2Args, ReduceSchemaArgNames,
           ReduceSchema2Attrs, ReduceHasher);
MNM_TVMJIT(Min, "mnm.op.min", ReduceArgs, ReduceSchema2Args, ReduceSchemaArgNames,
           ReduceSchema2Attrs, ReduceHasher);
MNM_TVMJIT(All, "mnm.op.all", ReduceArgs, ReduceSchema2Args, ReduceSchemaArgNames,
           ReduceSchema2Attrs, ReduceHasher);
MNM_TVMJIT(Any, "mnm.op.any", ReduceArgs, ReduceSchema2Args, ReduceSchemaArgNames,
           ReduceSchema2Attrs, ReduceHasher);
MNM_TVMJIT(Mean, "mnm.op.mean", ReduceArgs, ReduceSchema2Args, ReduceSchemaArgNames,
           ReduceSchema2Attrs, ReduceHasher);

std::vector<Value> ReduceDxSchema2Args(const ReduceDxArgs* args) {
  return {args->x, args->y, args->dy};
}

std::vector<std::string> ReduceDxSchemaArgNames() {
  return {"x", "y", "dy"};
}

Attrs ReduceDxSchema2Attrs(const ReduceDxArgs* args) {
  auto attrs = make_object<tvm_attrs::ReduceAttrs>();
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
  attrs->exclude = false;
  return Attrs(attrs);
}

HashKey ReduceDxHasher(const std::vector<Type>& param_types, const Type& ret_type,
                       const ReduceDxArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, ret_type, nullptr);
  for (int i = 0, n = args->axis.size(); i < n; ++i) {
    key << args->axis[i];
  }
  key << args->keepdims;
  return key;
}

MNM_TVMJIT(MeanDx, "mnm.op.mean_dx", ReduceDxArgs, ReduceDxSchema2Args, ReduceDxSchemaArgNames,
           ReduceDxSchema2Attrs, ReduceDxHasher);  // NOLINT

}  // namespace tvmjit
}  // namespace op
}  // namespace mnm
