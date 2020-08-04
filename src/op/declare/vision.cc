/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/declare/vision.cc
 * \brief Declaration of vision-specific operators
 */
#include "mnm/op.h"
#include "mnm/tensor.h"
#include "../schema/vision.h"
#include "./declare_utils.h"

namespace mnm {
namespace op {
namespace declare {

using namespace mnm::op::schema;
using namespace mnm::value;

MNM_OP_DECLARE("mnm.op.get_valid_counts", [](const CallValues& call) {
  const auto* args = call->args.as<GetValidCountsArgs>();
  CHECK(args != nullptr);
  DLTensor* data = args->data;
  CHECK_EQ(data->ndim, 3) << "Input data should be 3-D.";

  std::vector<TensorValue> ret;
  std::vector<int64_t> oshape(data->shape, data->shape + 1);
  std::vector<int64_t> data_shape(data->shape, data->shape + data->ndim);
  std::vector<int64_t> oshape_indices(data->shape, data->shape + 2);
  ret.push_back(TensorValue::Assemble(data->ctx, DType(DTypeCode::kInt(), 32), oshape));
  ret.push_back(TensorValue::Assemble(data->ctx, data->dtype, data_shape));
  ret.push_back(TensorValue::Assemble(data->ctx, DType(DTypeCode::kInt(), 32), oshape_indices));

  call->out = TupleValue::make(ir::Array<Value>(ret.begin(), ret.end()));
  call->ctx = data->ctx;
}).set_attr<TOpPattern>("TOpPattern", kInjective);

}  // namespace declare
}  // namespace op
}  // namespace mnm
