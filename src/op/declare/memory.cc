/*!
 * Copyright (c) 2021 by Contributors
 * \file src/op/declare/memory.cc
 * \brief Declaration of memory-related operators.
 */
#include <string>
#include "mnm/op.h"
#include "mnm/tensor.h"
#include "mnm/op_utils.h"
#include "../schema/memory.h"
#include "../../common/shape_utils.h"

namespace mnm {
namespace op {
namespace declare {

using namespace mnm::op::schema;
using namespace mnm::value;
using namespace mnm::common::shape_utils;

MNM_OP_DECLARE("mnm.op.device_copy", [](const CallValues& call) {
  const auto* args = call->args.as<DeviceCopyArgs>();
  CHECK(args != nullptr);
  DLTensor* data = args->data;
  std::vector<int64_t> shape(data->shape, data->shape + data->ndim);
  CHECK_EQ(static_cast<int>(data->device.device_type), args->src_dev_type);
  DLDevice out_ctx;
  out_ctx.device_type = static_cast<DLDeviceType>(args->dst_dev_type);
  out_ctx.device_id = 0;
  call->out = TensorValue::Assemble(/*dev=*/out_ctx,
                                    /*dtype=*/data->dtype,
                                    /*shape=*/shape);
  call->device = data->device;
}).set_attr<TOpPattern>("TOpPattern", kOpaque);

MNM_OP_DECLARE("mnm.op.fuse_tensor", [](const CallValues& call) {
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

MNM_OP_DECLARE("mnm.op.defuse_tensor", [](const CallValues& call) {
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
}  // namespace mnm
