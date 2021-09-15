/*!
 * Copyright (c) 2020 by Contributors
 * \file src/op/ty/collective_comm.cc
 * \brief Typing of collective communicate operators
 */
#include <tvm/relay/type.h>
#include <tvm/tir/op.h>
#include "mnm/type.h"
#include "../schema/communication.h"
#include "./utils.h"

namespace mnm {
namespace op {

using namespace mnm::ir;
using namespace mnm::value;
using namespace mnm::op::schema;

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

MNM_OP_TYPE("mnm.op._allreduce", "NCCLAllReduce", IdentityType<AllreduceArgs>);
MNM_OP_TYPE("mnm.op._broadcast", "NCCLBroadcast", IdentityType<BroadcastArgs>);
MNM_OP_TYPE("mnm.op._reduce", "NCCLReduce", IdentityType<CommReduceArgs>);

Type ReduceScatterInfer(const CallValues& value) {
  const auto* args = value->args.as<ReduceScatterArgs>();
  CHECK(args != nullptr);
  CHECK_GE(args->x.size(), 1U);
  const auto& ty = GetType(args->x[0]);
  for (const auto& x : args->x) {
    CHECK(GetType(x) == ty);
  }
  return ty;
}

MNM_OP_TYPE("mnm.op._reduce_scatter", "NCCLReduceScatter", ReduceScatterInfer);

Type SendInfer(const CallValues& value) {
  const auto* args = value->args.as<SendArgs>();
  CHECK(args != nullptr);
  const auto& ty = Downcast<TensorType>(GetType(args->x));
  return TensorType({}, ty->dtype);
}

MNM_OP_TYPE("mnm.op._send", "NCCLSend", SendInfer);

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

MNM_OP_TYPE("mnm.op._recv", "NCCLRecv", RecvInfer);

}  // namespace op
}  // namespace mnm
