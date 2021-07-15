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
  ret.push_back(TensorValue::Assemble(data->device, DType(DTypeCode::kInt(), 32), oshape));
  ret.push_back(TensorValue::Assemble(data->device, data->dtype, data_shape));
  ret.push_back(TensorValue::Assemble(data->device, DType(DTypeCode::kInt(), 32), oshape_indices));

  call->out = TupleValue::make(ir::Array<Value>(ret.begin(), ret.end()));
  call->device = data->device;
});

MNM_OP_DECLARE("mnm.op.non_max_suppression", [](const CallValues& call) {
  const auto* args = call->args.as<NonMaxSuppressionArgs>();
  CHECK(args != nullptr);
  DLTensor* data = args->data;
  DLTensor* valid_count = args->valid_count;
  CHECK_EQ(data->ndim, 3) << "Input data should be 3-D.";
  CHECK_EQ(valid_count->ndim, 1) << "Input valid count should be 1-D.";

  if (args->return_indices) {
    std::vector<TensorValue> ret;
    std::vector<int64_t> oshape(data->shape, data->shape + 2);
    std::vector<int64_t> count_shape({*data->shape, 1});
    ret.push_back(TensorValue::Assemble(data->device, DType(DTypeCode::kInt(), 32), oshape));
    ret.push_back(TensorValue::Assemble(data->device, DType(DTypeCode::kInt(), 32), count_shape));
    call->out = TupleValue::make(ir::Array<Value>(ret.begin(), ret.end()));
  } else {
    std::vector<int64_t> dshape(data->shape, data->shape + data->ndim);
    call->out = TensorValue::Assemble(/*dev=*/data->device,
                                      /*dtype=*/data->dtype,
                                      /*shape=*/dshape);
  }
  call->device = data->device;
});

MNM_OP_DECLARE(
    "mnm.op.roi_align", ([](const CallValues& call) {
      const auto* args = call->args.as<RoiAlignArgs>();
      CHECK(args != nullptr);
      DLTensor* data = args->data;
      DLTensor* rois = args->rois;
      CHECK_EQ(data->ndim, 4) << "Input data should be 4-D.";
      CHECK_EQ(rois->ndim, 2) << "Input rois should be 2-D.";
      // assign output type
      std::vector<int64_t> oshape;
      if (args->layout == "NCHW") {
        oshape = {rois->shape[0], data->shape[1], args->pooled_size[0], args->pooled_size[1]};
      } else {
        ICHECK_EQ(args->layout, "NHWC") << "Unexpected ROI Align layout";
        oshape = {rois->shape[0], args->pooled_size[0], args->pooled_size[1], data->shape[3]};
      }
      call->device = data->device;
      call->out = TensorValue::Assemble(/*dev=*/data->device,
                                        /*dtype=*/data->dtype,
                                        /*shape=*/oshape);
    }));

MNM_OP_DECLARE("mnm.op.roi_align_dx", [](const CallValues& call) {
  const auto* args = call->args.as<RoiAlignDxArgs>();
  CHECK(args != nullptr);
  DLTensor* data = args->data;
  std::vector<int64_t> dshape(data->shape, data->shape + data->ndim);
  call->device = data->device;
  call->out = TensorValue::Assemble(/*dev=*/data->device,
                                    /*dtype=*/data->dtype,
                                    /*shape=*/dshape);
});

}  // namespace declare
}  // namespace op
}  // namespace mnm
