/*!
 * Copyright (c) 2020 by Contributors
 * Auto generated. Do not touch.
 * \file src/op/schema/stream.h
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
class StreamArgs : public ir::AttrsNode<StreamArgs> {
 public:
  value::BaseTensorValue x;
  int stream_tag{0};
  MNM_OP_SCHEMA(StreamArgs, "mnm.args.stream");
};
}  // namespace schema
}  // namespace op
}  // namespace mnm
