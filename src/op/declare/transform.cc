/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/declare/transform.cc
 * \brief Declaration of transform operators
 */

#include <functional>
#include <numeric>

#include "mnm/op.h"
#include "mnm/tensor.h"
#include "../schema/ufunc.h"
#include "../schema/likes.h"
#include "../../common/shape_utils.h"

namespace mnm {
namespace op {
namespace declare {

using namespace mnm::op::schema;
using namespace mnm::value;
using common::shape_utils::IsCompact;
using tensor::Tensor;

MNM_OP_DECLARE("mnm.op.batch_flatten", [](const CallValues& call) {
  const auto* args = call->args.as<UnaryArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->x;
  const int ndim = x->ndim;
  CHECK_GE(ndim, 2) << "ValueError: batch_flatten only works with ndim >= 2";

  if (IsCompact(*x)) {
    const int64_t* dshape = x->shape;
    int64_t flat{1};
    for (int i = 1; i < ndim; ++i) {
      flat = flat * int64_t{dshape[i]};
    }
    call->callee = ir::NullValue<OpValue>();
    call->out = TensorValue::make(Tensor(args->x).CreateView({dshape[0], flat}, {}, nullptr));
    return;
  }
  LOG(FATAL) << "NotImplementedError: for now we only support batch_flatten on contiguous tensor.";
  throw;
});

MNM_OP_DECLARE("mnm.op.reshape", [](const CallValues &call) {
  const auto* args = call->args.as<ReshapeArgs>();
  DLTensor *x = args->x;
  const std::vector<int64_t> &shape = args->shape;
  call->ctx = x->ctx;
  call->callee = ir::NullValue<OpValue>();
  if (IsCompact(*x)) {
    int64_t origin = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
    int64_t reshaped = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
    CHECK_EQ(origin, reshaped) << "Number of elements mismatch after reshaping!";
    call->out = TensorValue::make(Tensor(args->x).CreateView(shape));
    return;
  }
  LOG(FATAL) << "NotImplementedError: for now we only support reshape on contiguous tensor.";
  throw;
});

}  // namespace declare
}  // namespace op
}  // namespace mnm
