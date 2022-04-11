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
#include "raf/dist_context.h"
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
RAF_OP_TYPE("raf.op._broadcast", "NCCLBroadcast", IdentityType<BroadcastArgs>);
RAF_OP_TYPE("raf.op._reduce", "NCCLReduce", IdentityType<CommReduceArgs>);

Type ReduceScatterInfer(const CallValues& value) {
  static auto* structural_equal = tvm::runtime::Registry::Get("node.StructuralEqual");
  ICHECK(structural_equal) << "node.StructuralEqual is not registered.";

  const auto* args = value->args.as<ReduceScatterArgs>();
  CHECK(args != nullptr);
  CHECK_GE(args->x.size(), 1U);
  const auto& ty = GetType(args->x[0]);
  for (const auto& x : args->x) {
    (*structural_equal)(GetType(x), ty, true, true);
  }
  return ty;
}

RAF_OP_TYPE("raf.op._reduce_scatter", "NCCLReduceScatter", ReduceScatterInfer);

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
  auto size = Communicator::Get(args->rank_list)->size;
  auto ttype = GetType(args->x).as<TensorTypeNode>();
  auto shape = ttype->shape;
  auto new_size = shape[args->axis].as<IntImmNode>()->value * size;
  shape.Set(args->axis, Integer(new_size));
  return TensorType(shape, DataType(ttype->dtype));
}

RAF_OP_TYPE("raf.op._allgather", "NCCLAllGather", AllGatherInfer);

}  // namespace op
}  // namespace raf
