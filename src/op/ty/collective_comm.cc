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
namespace type {

using namespace mnm::value;
using namespace mnm::op::schema;
using namespace tvm;
using namespace tvm::relay;

Type AllReduceInfer(const CallValues& value) {
  const auto* args = value->args.as<AllreduceArgs>();
  CHECK(args != nullptr);
  CHECK(args->x.size() > 0);
  Array<Type> x;
  std::transform(args->x.begin(), args->x.end(), std::back_inserter(x), GetType);
  return TupleType(x);
}

MNM_OP_TYPE("mnm.op._allreduce", "AllReduce", AllReduceInfer);

Type StreamInfer(const CallValues& value) {
  const auto* args = value->args.as<StreamControlArgs>();
  CHECK(args != nullptr);
  return GetType(args->x);
}

MNM_OP_TYPE("mnm.op.stream_sync", "StreamSync", StreamInfer);

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

MNM_OP_TYPE("mnm.op._reduce_scatter", "ReduceScatter", ReduceScatterInfer);

}  // namespace type
}  // namespace op
}  // namespace mnm
