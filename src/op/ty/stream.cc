/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/ty/stream.cc
 * \brief Typing relations of cuda stream operators
 */
#include <tvm/relay/type.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/env_func.h>
#include "raf/type.h"
#include "../schema/stream.h"
#include "./utils.h"

namespace raf {
namespace op {

using namespace raf::ir;
using namespace raf::value;
using namespace raf::op::schema;

Type StreamEventInfer(const CallValues& value) {
  return TupleType::Empty();
}

Type StreamInfer(const CallValues& value) {
  const auto* args = value->args.as<StreamArgs>();
  CHECK(args != nullptr);
  return GetType(args->x);
}

RAF_OP_TYPE("raf.op.set_stream", "StreamSwitch", StreamEventInfer);
RAF_OP_TYPE("raf.op.add_event", "EventAdd", StreamEventInfer);
RAF_OP_TYPE("raf.op.wait_event", "EventWait", StreamEventInfer);
RAF_OP_TYPE("raf.op.stream_barrier", "StreamBarrier", StreamEventInfer);
RAF_OP_TYPE("raf.op.stream_sync", "StreamSync", StreamInfer);

}  // namespace op
}  // namespace raf
