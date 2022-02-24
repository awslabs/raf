/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/ty/vision.cc
 * \brief Typing of vision operators
 */
#include <tvm/relay/type.h>
#include "raf/type.h"
#include "../schema/vision.h"
#include "./utils.h"

namespace raf {
namespace op {
namespace type {

using namespace raf::ir;
using namespace raf::value;
using namespace raf::op::schema;

Type GetValidCountsInfer(const CallValues& value) {
  const auto* args = value->args.as<GetValidCountsArgs>();
  CHECK(args != nullptr);
  TensorType data = Downcast<TensorType>(GetType(args->data));
  CHECK_EQ(data->shape.size(), 3) << "ValueError: Input data should be 3-D";

  Array<Type> ret;
  Array<PrimExpr> oshape(data->shape.begin(), data->shape.begin() + 1);
  Array<PrimExpr> data_shape(data->shape);
  Array<PrimExpr> oshape_indices(data->shape.begin(), data->shape.begin() + 2);
  ret.push_back(TensorType(oshape, DataType::Int(32)));
  ret.push_back(TensorType(data_shape, data->dtype));
  ret.push_back(TensorType(oshape_indices, DataType::Int(32)));
  return TupleType(ret);
}

RAF_OP_TYPE("raf.op.get_valid_counts", "GetValidCounts", GetValidCountsInfer);

Type NonMaxSuppressionInfer(const CallValues& value) {
  const auto* args = value->args.as<NonMaxSuppressionArgs>();
  CHECK(args != nullptr);
  TensorType data = Downcast<TensorType>(GetType(args->data));
  TensorType valid_count = Downcast<TensorType>(GetType(args->valid_count));
  CHECK_EQ(data->shape.size(), 3) << "ValueError: Input data should be 3-D";
  CHECK_EQ(valid_count->shape.size(), 1) << "ValueError: Input valid count should be 1-D";

  if (args->return_indices) {
    Array<Type> ret;
    Array<PrimExpr> oshape(data->shape.begin(), data->shape.begin() + 2);
    Array<PrimExpr> count_shape({data->shape[0], 1});
    ret.push_back(TensorType(oshape, DataType::Int(32)));
    ret.push_back(TensorType(count_shape, DataType::Int(32)));
    return TupleType(ret);
  } else {
    return data;
  }
}

RAF_OP_TYPE("raf.op.non_max_suppression", "NonMaxSuppression", NonMaxSuppressionInfer);

Type RoiAlignInfer(const CallValues& value) {
  const auto* args = value->args.as<RoiAlignArgs>();
  CHECK(args != nullptr);
  TensorType data = Downcast<TensorType>(GetType(args->data));
  TensorType rois = Downcast<TensorType>(GetType(args->rois));
  CHECK_EQ(data->shape.size(), 4) << "Input data should be 4-D.";
  CHECK_EQ(rois->shape.size(), 2) << "Input rois should be 2-D.";
  // assign output type
  std::vector<PrimExpr> oshape;
  if (args->layout == "NCHW") {
    oshape.push_back(rois->shape[0]);
    oshape.push_back(data->shape[1]);
    oshape.push_back(int32_t(args->pooled_size[0]));
    oshape.push_back(int32_t(args->pooled_size[1]));
  } else {
    ICHECK_EQ(args->layout, "NHWC") << "Unexpected ROI Align layout " << args->layout;
    oshape.push_back(rois->shape[0]);
    oshape.push_back(int32_t(args->pooled_size[0]));
    oshape.push_back(int32_t(args->pooled_size[1]));
    oshape.push_back(data->shape[3]);
  }
  return TensorType(oshape, data->dtype);
}

RAF_OP_TYPE("raf.op.roi_align", "RoiAlign", RoiAlignInfer);

Type RoiAlignDxInfer(const CallValues& value) {
  const auto* args = value->args.as<RoiAlignDxArgs>();
  CHECK(args != nullptr);
  return Downcast<TensorType>(GetType(args->data));
}

RAF_OP_TYPE("raf.op.roi_align_dx", "RoiAlignDx", RoiAlignDxInfer);

}  // namespace type
}  // namespace op
}  // namespace raf
