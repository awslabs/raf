/*!
 * Copyright (c) 2021 by Contributors
 * \file src/op/ty/stream.cc
 * \brief Typing relations of cuda stream operators
 */
#include <tvm/relay/type.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/env_func.h>
#include "mnm/type.h"
#include "../schema/stream.h"
#include "./utils.h"

namespace mnm {
namespace op {

using namespace mnm::ir;
using namespace mnm::value;
using namespace mnm::op::schema;

Type StreamEventInfer(const CallValues& value) {
  return TupleType::Empty();
}

Type StreamInfer(const CallValues& value) {
  const auto* args = value->args.as<StreamArgs>();
  CHECK(args != nullptr);
  return GetType(args->x);
}

MNM_OP_TYPE("mnm.op.set_stream", "StreamSwitch", StreamEventInfer);
MNM_OP_TYPE("mnm.op.add_event", "EventAdd", StreamEventInfer);
MNM_OP_TYPE("mnm.op.wait_event", "EventWait", StreamEventInfer);
MNM_OP_TYPE("mnm.op.stream_barrier", "StreamBarrier", StreamEventInfer);
MNM_OP_TYPE("mnm.op.stream_sync", "StreamSync", StreamInfer);

}  // namespace op
}  // namespace mnm
