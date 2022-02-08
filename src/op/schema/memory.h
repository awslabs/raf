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
class DefuseTensorArgs : public ir::AttrsNode<DefuseTensorArgs> {
 public:
  value::BaseTensorValue data;
  std::vector<int64_t> sizes;
  std::vector<int64_t> shapes;
  std::vector<int64_t> shape_indices;
  MNM_OP_SCHEMA(DefuseTensorArgs, "mnm.args.defuse_tensor");
};

class DeviceCopyArgs : public ir::AttrsNode<DeviceCopyArgs> {
 public:
  value::BaseTensorValue data;
  std::string src_device{"cpu"};
  std::string dst_device{"cpu"};
  MNM_OP_SCHEMA(DeviceCopyArgs, "mnm.args.device_copy");
};

class FuseTensorArgs : public ir::AttrsNode<FuseTensorArgs> {
 public:
  std::vector<value::BaseTensorValue> data;
  MNM_OP_SCHEMA(FuseTensorArgs, "mnm.args.fuse_tensor");
};
}  // namespace schema
}  // namespace op
}  // namespace mnm
