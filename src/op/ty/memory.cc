/*!
 * Copyright (c) 2021 by Contributors
 * \file src/op/ty/memory.cc
 * \brief Typing of memory operators
 */

#include <tvm/relay/type.h>
#include "mnm/type.h"
#include "../schema/memory.h"
#include "./utils.h"
#include "../../common/shape_utils.h"

namespace mnm {
namespace op {

using namespace mnm::ir;
using namespace schema;
using namespace mnm::common::shape_utils;

Type DeviceCopyInfer(const CallValues& value) {
  const auto* args = value->args.as<DeviceCopyArgs>();
  CHECK(args != nullptr);
  return GetType(args->data);
}

MNM_OP_TYPE("mnm.op.device_copy", "Memory", DeviceCopyInfer);

Type FuseTensorInfer(const CallValues& value) {
  const auto* args = value->args.as<FuseTensorArgs>();
  CHECK(args != nullptr);
  DataType dtype = Downcast<TensorType>(GetType(args->data[0]))->dtype;
  size_t total_size = 0;
  for (int i = 0; i < args->data.size(); ++i) {
    total_size += GetSizeFromType(Downcast<TensorType>(GetType(args->data[i])));
  }
  return TensorType({Integer(total_size)}, dtype);
}

MNM_OP_TYPE("mnm.op.fuse_tensor", "FuseTensor", FuseTensorInfer);

Type DefuseTensorInfer(const CallValues& value) {
  const auto* args = value->args.as<DefuseTensorArgs>();
  CHECK(args != nullptr);
  DataType dtype = Downcast<TensorType>(GetType(args->data))->dtype;
  std::vector<int64_t> shapes = args->shapes;
  std::vector<int64_t> shape_indices = args->shape_indices;
  Array<Type> tuple_types;
  size_t index = 0;
  for (int i = 0; i < args->shape_indices.size(); ++i) {
    Array<IndexExpr> shape = {};
    for (auto it = args->shapes.begin() + index;
         it != args->shapes.begin() + args->shape_indices[i]; ++it) {
      shape.push_back(Integer(*it));
    }
    tuple_types.push_back(TensorType(shape, dtype));
    index = shape_indices[i];
  }
  return TupleType(tuple_types);
}

MNM_OP_TYPE("mnm.op.defuse_tensor", "DefuseTensor", DefuseTensorInfer);

}  // namespace op
}  // namespace mnm
