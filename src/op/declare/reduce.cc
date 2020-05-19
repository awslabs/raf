/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/declare/unary.cc
 * \brief Declaration of unary operators
 */
#include <numeric>
#include "mnm/op.h"
#include "mnm/tensor.h"
#include "../schema/reduce.h"

namespace mnm {
namespace op {
namespace declare {

using namespace mnm::op::schema;
using namespace mnm::value;

void Reduce(const CallValues &call) {
  const auto* args = call->args.as<ReduceArgs>();
  CHECK(args != nullptr);
  DLTensor *x = args->x;
  std::vector<int64_t> axis;
  auto ndim = x->ndim;
  if (args->axis.empty()) {
    axis.resize(ndim);
    std::iota(axis.begin(), axis.end(), 0);
  } else {
    axis = args->axis;
  }
  std::sort(axis.begin(), axis.end());
  axis.resize(std::unique(axis.begin(), axis.end()) - axis.begin());
  bool keepdims = args->keepdims;
  std::vector<int64_t> shape;
  if (keepdims) {
    for (int64_t i = 0; i < ndim; i++) {
      if (std::find(axis.begin(), axis.end(), i) != axis.end()) {
        shape.push_back(1);
      } else {
        shape.push_back(x->shape[i]);
      }
    }
  } else {
    for (int64_t i = 0; i < ndim; i++) {
      if (std::find(axis.begin(), axis.end(), i) == axis.end()) {
        shape.push_back(x->shape[i]);
      }
    }
  }
  call->ctx = x->ctx;
  call->out = TensorValue::Assemble(x->ctx, DType(DTypeCode::kInt(), 32), shape);
}

MNM_OP_DECLARE("mnm.op.argmax", Reduce);
MNM_OP_DECLARE("mnm.op.argmin", Reduce);

}  // namespace declare
}  // namespace op
}  // namespace mnm
