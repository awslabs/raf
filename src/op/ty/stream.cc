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

Type StreamInfer(const CallValues& value) {
  const auto* args = value->args.as<StreamArgs>();
  CHECK(args != nullptr);
  return GetType(args->x);
}

MNM_OP_TYPE("mnm.op.stream_sync", "StreamSync", StreamInfer);
MNM_OP_TYPE("mnm.op.stream_start", "Stream", StreamInfer);
MNM_OP_TYPE("mnm.op.stream_end", "Stream", StreamInfer);
MNM_OP_TYPE("mnm.op.stream_wait", "Stream", StreamInfer);

}  // namespace op
}  // namespace mnm
