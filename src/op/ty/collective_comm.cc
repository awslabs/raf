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

}  // namespace type
}  // namespace op
}  // namespace mnm
