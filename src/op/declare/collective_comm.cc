/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/declare/collective_comm.cc
 * \brief Declaration of collective communication operators
 */
#include "mnm/op.h"
#include "mnm/tensor.h"
#include "../schema/communication.h"
#include "../schema/ufunc.h"
#include "./declare_utils.h"

namespace mnm {
namespace op {
namespace declare {

using namespace mnm::op::schema;
using namespace mnm::value;
using tensor::Tensor;

void AllReduce(const CallValues& call) {
  const auto* args = call->args.as<AllreduceArgs>();
  CHECK(args != nullptr);
  ir::Array<Value> ret;
  auto& tv = args->x;
  const DLTensor* x = tv[0];
  call->device = x->device;
  for (int i = 0; i < tv.size(); ++i) {
    const DLTensor* x = tv[i];
    std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
    ret.push_back(TensorValue::Assemble(/*dev=*/x->device,
                                        /*dtype=*/x->dtype,
                                        /*shape=*/shape));
  }
  if (ret.size() == 0) call->callee = ir::NullValue<OpValue>();
  if (ret.size() == 1) {
    call->out = ret[0];
  } else {
    call->out = TupleValue::make(ir::Array<Value>(ret.begin(), ret.end()));
  }
}

MNM_OP_DECLARE("mnm.op._allreduce", AllReduce);

}  // namespace declare
}  // namespace op
}  // namespace mnm
