/*!
 * Copyright (c) 2021 by Contributors
 * \file src/op/ty/memory.cc
 * \brief Typing of memory operators
 */

#include <tvm/relay/type.h>
#include "mnm/type.h"
#include "../schema/memory.h"
#include "./utils.h"

namespace mnm {
namespace op {

using namespace mnm::ir;
using namespace schema;

Type DeviceCopyInfer(const CallValues& value) {
  const auto* args = value->args.as<DeviceCopyArgs>();
  CHECK(args != nullptr);
  return GetType(args->data);
}

MNM_OP_TYPE("mnm.op.device_copy", "Memory", DeviceCopyInfer);

}  // namespace op
}  // namespace mnm
