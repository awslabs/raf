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

void GenerateReduceShape(const ReduceArgs* args, const DLTensor* x, std::vector<int64_t>* shape) {
  CHECK(args != nullptr);
  auto ndim = x->ndim;
  std::vector<int64_t> axis;
  if (args->axis.empty()) {
    axis.resize(ndim);
    std::iota(axis.begin(), axis.end(), 0);
  } else {
    axis = args->axis;
  }
  std::sort(axis.begin(), axis.end());
  axis.resize(std::unique(axis.begin(), axis.end()) - axis.begin());

  bool exclude = args->exclude;
  if (exclude) {
    std::vector<int64_t> axis_exclude;
    for (int64_t i = 0; i < ndim; i++) {
      if (std::find(axis.begin(), axis.end(), i) == axis.end()) {
        axis_exclude.push_back(i);
      }
    }
    axis = axis_exclude;
  }
  bool keepdims = args->keepdims;
  if (keepdims) {
    for (int64_t i = 0; i < ndim; i++) {
      if (std::find(axis.begin(), axis.end(), i) != axis.end()) {
        shape->push_back(1);
      } else {
        shape->push_back(x->shape[i]);
      }
    }
  } else {
    for (int64_t i = 0; i < ndim; i++) {
      if (std::find(axis.begin(), axis.end(), i) == axis.end()) {
        shape->push_back(x->shape[i]);
      }
    }
  }
}

void ReduceOutInt(const CallValues& call) {
  const auto* args = call->args.as<ReduceArgs>();
  DLTensor* x = args->x;
  std::vector<int64_t> shape;
  GenerateReduceShape(args, x, &shape);
  call->device = x->device;
  call->out = TensorValue::Assemble(x->device, DType(DTypeCode::kInt(), 32), shape);
}

void ReduceOutSame(const CallValues& call) {
  const auto* args = call->args.as<ReduceArgs>();
  DLTensor* x = args->x;
  std::vector<int64_t> shape;
  GenerateReduceShape(args, x, &shape);
  call->device = x->device;
  call->out = TensorValue::Assemble(x->device, x->dtype, shape);
}

void ReduceDxOutSame(const CallValues& call) {
  // the shape of the output of reduce_dx op is same as input x
  const auto* args = call->args.as<ReduceDxArgs>();
  CHECK(args != nullptr);
  DLTensor* x = args->x;
  std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
  call->device = x->device;
  call->out = TensorValue::Assemble(/*dev=*/x->device,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/shape);
}

void MeanDxDecl(const CallValues& call) {
  const auto* args = call->args.as<MeanDxArgs>();
  CHECK(args != nullptr);
  DLTensor* dy = args->dy;
  std::vector<int64_t> shape = args->x_shape;
  call->device = dy->device;
  call->out = TensorValue::Assemble(/*dev=*/dy->device,
                                    /*dtype=*/dy->dtype,
                                    /*shape=*/shape);
}

MNM_OP_DECLARE("mnm.op.argmax", ReduceOutInt);
MNM_OP_DECLARE("mnm.op.argmin", ReduceOutInt);
MNM_OP_DECLARE("mnm.op.max", ReduceOutSame);
MNM_OP_DECLARE("mnm.op.min", ReduceOutSame);
MNM_OP_DECLARE("mnm.op.all", ReduceOutSame);
MNM_OP_DECLARE("mnm.op.any", ReduceOutSame);
MNM_OP_DECLARE("mnm.op.mean", ReduceOutSame);
MNM_OP_DECLARE("mnm.op.prod", ReduceOutSame);
MNM_OP_DECLARE("mnm.op.mean_dx", MeanDxDecl);
MNM_OP_DECLARE("mnm.op.prod_dx", ReduceDxOutSame);

}  // namespace declare
}  // namespace op
}  // namespace mnm
