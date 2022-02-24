/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/declare/memory.cc
 * \brief Declaration of memory-related operators.
 */
#include <string>
#include "raf/op.h"
#include "raf/tensor.h"
#include "raf/op_utils.h"
#include "../schema/memory.h"
#include "../../common/shape_utils.h"

namespace raf {
namespace op {
namespace declare {

using namespace raf::op::schema;
using namespace raf::value;
using namespace raf::common::shape_utils;

RAF_OP_DECLARE("raf.op.device_copy", [](const CallValues& call) {
  const auto* args = call->args.as<DeviceCopyArgs>();
  CHECK(args != nullptr);
  DLTensor* data = args->data;
  std::vector<int64_t> shape(data->shape, data->shape + data->ndim);

  const auto* str2dev = tvm::runtime::Registry::Get("raf._core.core_utils.str2dev");
  auto data_device = Device((tvm::Device)(*str2dev)(data->device));
  auto src_device = Device((tvm::Device)(*str2dev)(args->src_device));
  auto dst_device = Device((tvm::Device)(*str2dev)(args->dst_device));
  CHECK(data_device == src_device);

  call->out = TensorValue::Assemble(/*dev=*/dst_device,
                                    /*dtype=*/data->dtype,
                                    /*shape=*/shape);
  call->device = dst_device;
}).set_attr<TOpPattern>("TOpPattern", kOpaque);

RAF_OP_DECLARE("raf.op.fuse_tensor", [](const CallValues& call) {
  const auto* args = call->args.as<FuseTensorArgs>();
  CHECK(args != nullptr);
  auto& tv = args->data;
  const DLTensor* x = tv[0];
  call->device = x->device;
  int64_t total_size = 0;
  for (int i = 0; i < tv.size(); ++i) {
    const DLTensor* x = tv[i];
    total_size += BytesCompactTensor(*x) / ((x->dtype.bits + 7) / 8);
  }
  if (tv.size() == 0) {
    call->callee = ir::NullValue<OpValue>();
  } else {
    std::vector<int64_t> shape = {total_size};
    ir::Array<Value> ret;
    call->out = TensorValue::Assemble(/*dev=*/x->device,
                                      /*dtype=*/x->dtype,
                                      /*shape=*/shape);
  }
}).set_attr<TOpPattern>("TOpPattern", kOpaque);

RAF_OP_DECLARE("raf.op.defuse_tensor", [](const CallValues& call) {
  const auto* args = call->args.as<DefuseTensorArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->data;
  const std::vector<int64_t>& sizes = args->sizes;
  const std::vector<int64_t>& shape_indices = args->shape_indices;
  const std::vector<int64_t>& shapes = args->shapes;
  call->device = x->device;
  ir::Array<Value> ret;
  size_t total_size = 0;
  for (int i = 0; i < sizes.size(); ++i) {
    total_size += sizes[i];
  }
  CHECK(total_size == (int64_t)(BytesCompactTensor(*x) / ((x->dtype.bits + 7) / 8)))
      << "Input tensor size should be " << total_size << ", got "
      << (int64_t)(BytesCompactTensor(*x) / ((x->dtype.bits + 7) / 8)) << ".";
  int64_t start_idx = 0;
  for (int i = 0; i < shape_indices.size(); ++i) {
    std::vector<int64_t> shape(shapes.begin() + start_idx, shapes.begin() + shape_indices[i]);
    ret.push_back(TensorValue::Assemble(/*dev=*/x->device,
                                        /*dtype=*/x->dtype,
                                        /*shape=*/shape));
    start_idx = shape_indices[i];
  }
  if (ret.size() == 0) {
    call->callee = ir::NullValue<OpValue>();
  } else {
    call->out = TupleValue::make(ret);
  }
}).set_attr<TOpPattern>("TOpPattern", kOpaque);

}  // namespace declare
}  // namespace op
}  // namespace raf
