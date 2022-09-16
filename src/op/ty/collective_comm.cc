/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/ty/collective_comm.cc
 * \brief Typing of collective communicate operators
 */
#include <tvm/relay/type.h>
#include <tvm/tir/op.h>
#include "raf/dist_config.h"
#include "raf/communicator.h"
#include "raf/type.h"
#include "../schema/communication.h"
#include "./utils.h"

namespace raf {
namespace op {

using namespace raf::ir;
using namespace raf::value;
using namespace raf::op::schema;
using namespace raf::distributed::communicator;

template <typename T>
Type IdentityType(const CallValues& value) {
  const auto* args = value->args.as<T>();
  CHECK(args != nullptr);
  CHECK(args->x.size() > 0);
  if (args->x.size() == 1) {
    return GetType(args->x[0]);
  }
  Array<Type> x;
  std::transform(args->x.begin(), args->x.end(), std::back_inserter(x), GetType);
  return TupleType(x);
}

RAF_OP_TYPE("raf.op._allreduce", "NCCLAllReduce", IdentityType<AllreduceArgs>);

Type CommGatherInfer(const CallValues& value) {
  const auto* args = value->args.as<GatherScatterArgs>();
  CHECK(args != nullptr);
  int size = GetGlobalCommunicator()->size;
  const auto& ty = GetType(args->x);
  auto tpn = ty.as<TensorTypeNode>();
  auto shape = tpn->shape;
  auto old_size = shape[0].as<IntImmNode>()->value;
  auto new_size = old_size * size;
  shape.Set(0, Integer(new_size));
  return TensorType(shape, DataType(tpn->dtype));
}

RAF_OP_TYPE("raf.op._gather", "NCCLGather", CommGatherInfer);

Type CommScatterInfer(const CallValues& value) {
  const auto* args = value->args.as<GatherScatterArgs>();
  CHECK(args != nullptr);
  int size = GetGlobalCommunicator()->size;
  const auto& ty = GetType(args->x);
  auto tpn = ty.as<TensorTypeNode>();
  auto shape = tpn->shape;
  auto old_size = shape[0].as<IntImmNode>()->value;
  CHECK(old_size % size == 0);
  auto new_size = old_size / size;
  shape.Set(0, Integer(new_size));
  return TensorType(shape, DataType(tpn->dtype));
}

RAF_OP_TYPE("raf.op._scatter", "NCCLScatter", CommScatterInfer);

template <typename T>
Type TensorIdentityType(const CallValues& value) {
  const auto* args = value->args.as<T>();
  CHECK(args != nullptr);
  return GetType(args->x);
}

RAF_OP_TYPE("raf.op._all_to_all", "NCCLAllToAll", TensorIdentityType<AllToAllArgs>);
RAF_OP_TYPE("raf.op._broadcast", "NCCLBroadcast", TensorIdentityType<BroadcastArgs>);
RAF_OP_TYPE("raf.op._reduce", "NCCLReduce", TensorIdentityType<CommReduceArgs>);

Type ReduceScatterInfer(const CallValues& value) {
  static auto* structural_equal = tvm::runtime::Registry::Get("node.StructuralEqual");
  ICHECK(structural_equal) << "node.StructuralEqual is not registered.";

  const auto* args = value->args.as<ReduceScatterArgs>();
  CHECK(args != nullptr);
  const auto& ty = GetType(args->x);
  int size;
  if (args->rank_list.defined()) {
    size = Communicator::Get("void", args->rank_list)->size;
  } else {
    size = GetGlobalCommunicator()->size;
  }
  auto tpn = ty.as<TensorTypeNode>();
  auto shape = tpn->shape;
  auto old_size = shape[0].as<IntImmNode>()->value;
  CHECK(old_size % size == 0);
  auto new_size = old_size / size;
  shape.Set(0, Integer(new_size));
  return TensorType(shape, DataType(tpn->dtype));
}

RAF_OP_TYPE("raf.op._reduce_scatter", "NCCLReduceScatter", ReduceScatterInfer);

Type GroupReduceScatterInfer(const CallValues& value) {
  const auto* args = value->args.as<GroupReduceScatterArgs>();
  CHECK(args != nullptr);
  int size = GetGlobalCommunicator()->size;
  Array<Type> ret;
  for (const auto& tv : args->tensor_list) {
    const auto& ty = GetType(tv);
    auto tpn = ty.as<TensorTypeNode>();
    auto shape = tpn->shape;
    auto old_size = shape[0].as<IntImmNode>()->value;
    CHECK(old_size % size == 0);
    auto new_size = old_size / size;
    shape.Set(0, Integer(new_size));
    ret.push_back(TensorType(shape, DataType(tpn->dtype)));
  }
  return TupleType(ret);
}

RAF_OP_TYPE("raf.op._group_reduce_scatter", "NCCLGroupReduceScatter", GroupReduceScatterInfer);

Type SendInfer(const CallValues& value) {
  const auto* args = value->args.as<SendArgs>();
  CHECK(args != nullptr);
  const auto& ty = Downcast<TensorType>(GetType(args->x));
  return TensorType({}, ty->dtype);
}

RAF_OP_TYPE("raf.op._send", "NCCLSend", SendInfer);

Type RecvInfer(const CallValues& value) {
  const auto* args = value->args.as<RecvArgs>();
  CHECK(args != nullptr);
  std::string dtype = args->dtype;
  Array<PrimExpr> shape;
  for (const auto& s : args->shape) {
    shape.push_back(Integer(s));
  }
  return TensorType(shape, DataType(ir::String2DLDataType(args->dtype)));
}

RAF_OP_TYPE("raf.op._recv", "NCCLRecv", RecvInfer);

Type AllGatherInfer(const CallValues& value) {
  const auto* args = value->args.as<AllgatherArgs>();
  CHECK(args != nullptr);
  int size;
  if (args->rank_list.defined()) {
    size = Communicator::Get("void", args->rank_list)->size;
  } else {
    size = GetGlobalCommunicator()->size;
  }
  auto ttype = GetType(args->x).as<TensorTypeNode>();
  auto shape = ttype->shape;
  auto new_size = shape[args->axis].as<IntImmNode>()->value * size;
  shape.Set(args->axis, Integer(new_size));
  return TensorType(shape, DataType(ttype->dtype));
}

RAF_OP_TYPE("raf.op._allgather", "NCCLAllGather", AllGatherInfer);

Type GroupAllGatherInfer(const CallValues& value) {
  const auto* args = value->args.as<GroupAllgatherArgs>();
  CHECK(args != nullptr);
  int size = GetGlobalCommunicator()->size;
  Array<Type> ret;
  for (const auto& tv : args->tensor_list) {
    auto ttype = GetType(tv).as<TensorTypeNode>();
    auto shape = ttype->shape;
    auto new_size = shape[args->axis].as<IntImmNode>()->value * size;
    shape.Set(args->axis, Integer(new_size));
    ret.push_back(TensorType(shape, DataType(ttype->dtype)));
  }
  return TupleType(ret);
}

RAF_OP_TYPE("raf.op._group_allgather", "NCCLGroupAllGather", GroupAllGatherInfer);

}  // namespace op
}  // namespace raf
