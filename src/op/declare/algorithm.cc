/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/declare/vision.cc
 * \brief Declaration of algorithm-specific operators
 */
#include "mnm/op.h"
#include "mnm/tensor.h"
#include "../schema/algorithm.h"
#include "./declare_utils.h"

namespace mnm {
namespace op {
namespace declare {

using namespace mnm::op::schema;
using namespace mnm::value;

MNM_OP_DECLARE("mnm.op.argsort", [](const CallValues& call) {
  const auto* args = call->args.as<ArgsortArgs>();
  CHECK(args != nullptr);
  DLTensor* data = args->data;
  std::vector<int64_t> oshape(data->shape, data->shape + data->ndim);
  std::string dtype = args->dtype;

  call->out = TensorValue::Assemble(/*ctx=*/data->ctx,
                                    /*dtype=*/ir::String2DLDataType(dtype),
                                    /*shape=*/oshape);
  call->device = data->ctx;
}).set_attr<TOpPattern>("TOpPattern", kInjective);

MNM_OP_DECLARE("mnm.op.sort", [](const CallValues& call) {
  const auto* args = call->args.as<SortArgs>();
  CHECK(args != nullptr);
  DLTensor* data = args->data;
  std::vector<int64_t> oshape(data->shape, data->shape + data->ndim);

  call->out = TensorValue::Assemble(/*ctx=*/data->ctx,
                                    /*dtype=*/data->dtype,
                                    /*shape=*/oshape);
  call->device = data->ctx;
}).set_attr<TOpPattern>("TOpPattern", kInjective);

}  // namespace declare
}  // namespace op
}  // namespace mnm
