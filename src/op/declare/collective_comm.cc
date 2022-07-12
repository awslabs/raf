/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/declare/collective_comm.cc
 * \brief Declaration of collective communication operators
 */
#include "raf/op.h"
#include "raf/tensor.h"
#include "raf/communicator.h"
#include "../schema/communication.h"
#include "../schema/ufunc.h"
#include "./declare_utils.h"

namespace raf {
namespace op {
namespace declare {

using namespace raf::op::schema;
using namespace raf::value;
using namespace raf::distributed::communicator;
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

RAF_OP_DECLARE("raf.op._allreduce", AllReduce)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TRAFCollective>("TRAFCollective", true)
    .set_attr<TRAFInplaceUpdate>("TRAFInplaceUpdate", {{0, 0}});

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

RAF_OP_DECLARE("raf.op._reduce", Reduce)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TRAFCollective>("TRAFCollective", true)
    .set_attr<TRAFInplaceUpdate>("TRAFInplaceUpdate", {{0, 0}});

void AllGather(const CallValues& call) {
  const auto* args = call->args.as<AllgatherArgs>();
  CHECK(args != nullptr);
  ir::Array<Value> ret;
  const DLTensor* x = args->x;
  std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
  if (args->rank_list.defined()) {
    shape[args->axis] *= Communicator::Get("void", args->rank_list)->size;
  } else {
    shape[args->axis] *= GetGlobalCommunicator()->size;
  }
  call->device = x->device;
  call->out = TensorValue::Assemble(/*ctx=*/x->device,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/shape);
}

RAF_OP_DECLARE("raf.op._allgather", AllGather)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TRAFCollective>("TRAFCollective", true);

void GroupAllGather(const CallValues& call) {
  const auto* args = call->args.as<GroupAllgatherArgs>();
  CHECK(args != nullptr);
  std::vector<TensorValue> ret;
  const DLTensor* first_tensor = args->tensor_list[0];
  for (int i = 0; i < args->tensor_list.size(); ++i) {
    const DLTensor* x = args->tensor_list[i];
    std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
    shape[args->axis] *= GetGlobalCommunicator()->size;
    ret.push_back(TensorValue::Assemble(/*dev=*/x->device,
                                        /*dtype=*/x->dtype,
                                        /*shape=*/shape));
  }
  call->device = first_tensor->device;
  call->out = TupleValue::make(ir::Array<Value>(ret.begin(), ret.end()));
}

RAF_OP_DECLARE("raf.op._group_allgather", GroupAllGather)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TRAFCollective>("TRAFCollective", true)
    .set_attr<TRAFInplaceUpdate>("TRAFInplaceUpdate", {{2, 0}});

void ReduceScatter(const CallValues& call) {
  const auto* args = call->args.as<ReduceScatterArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->x;
  std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
  int size = 0;
  if (args->rank_list.defined()) {
    int rank = GetGlobalCommunicator()->rank;
    for (int i = 0; i < args->rank_list.size(); ++i) {
      for (int j = 0; j < args->rank_list[i].size(); ++j) {
        if (args->rank_list[i][j] == rank) {
          size = args->rank_list[i].size();
          break;
        }
      }
      if (size != 0) {
        break;
      }
    }
    if (size == 0) {
      size = 1;
    }
  }
  else {
    size = GetGlobalCommunicator()->size;
  }
  CHECK(shape[0] % size == 0);
  shape[0] = shape[0] / size;
  call->device = x->device;
  call->out = TensorValue::Assemble(/*ctx=*/x->device,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/shape);
}

RAF_OP_DECLARE("raf.op._reduce_scatter", ReduceScatter)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TRAFCollective>("TRAFCollective", true);

void GroupReduceScatter(const CallValues& call) {
  const auto* args = call->args.as<GroupReduceScatterArgs>();
  CHECK(args != nullptr);
  std::vector<BaseTensorValue> tvs = args->tensor_list;
  const DLTensor* first_tensor = tvs[0];
  std::vector<TensorValue> ret;
  int size = GetGlobalCommunicator()->size;
  for (const auto& tv : tvs) {
    const DLTensor* x = tv;
    std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
    CHECK(shape[0] % size == 0);
    shape[0] = shape[0] / size;
    ret.push_back(TensorValue::Assemble(/*dev=*/x->device,
                                        /*dtype=*/x->dtype,
                                        /*shape=*/shape));
  }
  call->device = first_tensor->device;
  call->out = TupleValue::make(ir::Array<Value>(ret.begin(), ret.end()));
}

RAF_OP_DECLARE("raf.op._group_reduce_scatter", GroupReduceScatter)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TRAFCollective>("TRAFCollective", true);

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

RAF_OP_DECLARE("raf.op._broadcast", Broadcast)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TRAFCollective>("TRAFCollective", true);

void Send(const CallValues& call) {
  const auto* args = call->args.as<SendArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->x;
  call->device = x->device;
  call->out = TensorValue::Assemble(/*ctx=*/x->device,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/std::vector<int64_t>{});
}

RAF_OP_DECLARE("raf.op._send", Send)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TRAFCollective>("TRAFCollective", true);

void Recv(const CallValues& call) {
  const auto* args = call->args.as<RecvArgs>();
  CHECK(args != nullptr);
  Device dev(DevType::kCUDA(), GetGlobalCommunicator()->rank);
  call->device = dev;
  call->out = TensorValue::Assemble(/*ctx=*/dev,
                                    /*dtype=*/ir::String2DLDataType(args->dtype),
                                    /*shape=*/args->shape);
}

RAF_OP_DECLARE("raf.op._recv", Recv)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TRAFCollective>("TRAFCollective", true);

}  // namespace declare
}  // namespace op
}  // namespace raf
