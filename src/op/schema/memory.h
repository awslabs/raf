/*!
 * Copyright (c) 2020 by Contributors
 * Auto generated. Do not touch.
 * \file src/op/schema/memory.h
 * \brief Operator schema.
 */
#pragma once
#include <vector>
#include <string>
#include "mnm/op.h"
#include "mnm/value.h"
namespace mnm {
namespace op {
namespace schema {
class DeviceCopyArgs : public ir::AttrsNode<DeviceCopyArgs> {
 public:
  value::BaseTensorValue data;
  int src_dev_type{0};
  int dst_dev_type{0};
  MNM_OP_SCHEMA(DeviceCopyArgs, "mnm.args.device_copy");
};
}  // namespace schema
}  // namespace op
}  // namespace mnm
