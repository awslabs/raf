/*!
 * Copyright (c) 2020 by Contributors
 * \file src/op/ty/vision.cc
 * \brief Typing of vision operators
 */
#include <tvm/relay/type.h>
#include "mnm/type.h"
#include "../schema/vision.h"
#include "./utils.h"

namespace mnm {
namespace op {
namespace type {

using namespace mnm::value;
using namespace schema;
using tvm::relay::Type;
using namespace tvm;
using namespace tvm::relay;

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

MNM_OP_TYPE("mnm.op.get_valid_counts", "GetValidCounts", GetValidCountsInfer);

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

MNM_OP_TYPE("mnm.op.non_max_suppression", "NonMaxSuppression", NonMaxSuppressionInfer);

}  // namespace type
}  // namespace op
}  // namespace mnm
