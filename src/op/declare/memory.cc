/*!
 * Copyright (c) 2021 by Contributors
 * \file src/op/declare/memory.cc
 * \brief Declaration of memory-related operators.
 */
#include "mnm/op.h"
#include "mnm/tensor.h"
#include "../schema/memory.h"

namespace mnm {
namespace op {
namespace declare {

using namespace mnm::op::schema;
using namespace mnm::value;

MNM_OP_DECLARE("mnm.op.device_copy", [](const CallValues& call) {
  const auto* args = call->args.as<DeviceCopyArgs>();
  CHECK(args != nullptr);
  DLTensor* data = args->data;
  std::vector<int64_t> shape(data->shape, data->shape + data->ndim);
  CHECK_EQ(static_cast<int>(data->ctx.device_type), args->src_dev_type);
  DLContext out_ctx;
  out_ctx.device_type = static_cast<DLDeviceType>(args->dst_dev_type);
  out_ctx.device_id = 0;
  call->out = TensorValue::Assemble(/*ctx=*/out_ctx,
                                    /*dtype=*/data->dtype,
                                    /*shape=*/shape);
  call->device = data->ctx;
}).set_attr<TOpPattern>("TOpPattern", kOpaque);

}  // namespace declare
}  // namespace op
}  // namespace mnm
