/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/declare/collective_comm.cc
 * \brief Declaration of collective communication operators
 */
#include "mnm/op.h"
#include "mnm/tensor.h"
#include "mnm/communicator.h"
#include "../schema/communication.h"
#include "../schema/ufunc.h"
#include "./declare_utils.h"

namespace mnm {
namespace op {
namespace declare {

using namespace mnm::op::schema;
using namespace mnm::value;
using namespace mnm::distributed::communicator;
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

MNM_OP_DECLARE("mnm.op._allreduce", AllReduce).set_attr<TOpPattern>("TOpPattern", kOpaque);

void Reduce(const CallValues& call) {
  const auto* args = call->args.as<CommReduceArgs>();
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
  if (ret.size() == 0) {
    call->callee = ir::NullValue<OpValue>();
  } else if (ret.size() == 1) {
    call->out = ret[0];
  } else {
    call->out = TupleValue::make(ir::Array<Value>(ret.begin(), ret.end()));
  }
}

MNM_OP_DECLARE("mnm.op._reduce", Reduce).set_attr<TOpPattern>("TOpPattern", kOpaque);

void AllGather(const CallValues& call) {
  const auto* args = call->args.as<AllgatherArgs>();
  CHECK(args != nullptr);
  ir::Array<Value> ret;
  const DLTensor* x = args->x;
  std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
  shape[args->axis] *= CommunicatorManager::Get()->GetCommunicator()->GetSize();
  call->device = x->device;
  call->out = TensorValue::Assemble(/*ctx=*/x->device,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/shape);
}

MNM_OP_DECLARE("mnm.op._allgather", AllGather).set_attr<TOpPattern>("TOpPattern", kOpaque);

void ReduceScatter(const CallValues& call) {
  const auto* args = call->args.as<ReduceScatterArgs>();
  CHECK(args != nullptr);
  std::vector<BaseTensorValue> tvs = args->x;
  CHECK_GE(tvs.size(), 1U);
  const DLTensor* x = tvs[0];
  std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
  for (const auto& tv : tvs) {
    const DLTensor* x = tv;
    CHECK(shape == std::vector<int64_t>(x->shape, x->shape + x->ndim));
  }
  call->device = x->device;
  call->out = TensorValue::Assemble(/*ctx=*/x->device,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/shape);
}

MNM_OP_DECLARE("mnm.op._reduce_scatter", ReduceScatter).set_attr<TOpPattern>("TOpPattern", kOpaque);

void Broadcast(const CallValues& call) {
  const auto* args = call->args.as<BroadcastArgs>();
  CHECK(args != nullptr);
  ir::Array<Value> ret;
  auto& tv = args->x;
  for (int i = 0; i < tv.size(); ++i) {
    const DLTensor* x = tv[i];
    std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
    ret.push_back(TensorValue::Assemble(/*dev=*/x->device,
                                        /*dtype=*/x->dtype,
                                        /*shape=*/shape));
  }
  if (ret.size() == 0) call->callee = ir::NullValue<OpValue>();
  const DLTensor* x = tv[0];
  call->device = x->device;
  if (ret.size() == 1) {
    call->out = ret[0];
  } else {
    call->out = TupleValue::make(ir::Array<Value>(ret.begin(), ret.end()));
  }
}

MNM_OP_DECLARE("mnm.op._broadcast", Broadcast).set_attr<TOpPattern>("TOpPattern", kOpaque);

void Send(const CallValues& call) {
  const auto* args = call->args.as<SendArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->x;
  call->device = x->device;
  call->out = TensorValue::Assemble(/*ctx=*/x->device,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/std::vector<int64_t>{});
}

MNM_OP_DECLARE("mnm.op._send", Send).set_attr<TOpPattern>("TOpPattern", kOpaque);

void Recv(const CallValues& call) {
  const auto* args = call->args.as<RecvArgs>();
  CHECK(args != nullptr);
  Device dev(DevType::kCUDA(), CommunicatorManager::Get()->GetCommunicator()->GetRank());
  call->device = dev;
  call->out = TensorValue::Assemble(/*ctx=*/dev,
                                    /*dtype=*/ir::String2DLDataType(args->dtype),
                                    /*shape=*/args->shape);
}

MNM_OP_DECLARE("mnm.op._recv", Recv).set_attr<TOpPattern>("TOpPattern", kOpaque);

}  // namespace declare
}  // namespace op
}  // namespace mnm
