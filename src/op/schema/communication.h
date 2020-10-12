/*!
 * Copyright (c) 2020 by Contributors
 * Auto generated. Do not touch.
 * \file src/op/schema/communication.h
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
class AllreduceArgs : public ir::AttrsNode<AllreduceArgs> {
 public:
  std::vector<value::BaseTensorValue> x;
  MNM_OP_SCHEMA(AllreduceArgs, "mnm.args._allreduce");
};

class StreamControlArgs : public ir::AttrsNode<StreamControlArgs> {
 public:
  value::BaseTensorValue x;
  int64_t stream_tag{0};
  MNM_OP_SCHEMA(StreamControlArgs, "mnm.args.stream_control");
};
}  // namespace schema
}  // namespace op
}  // namespace mnm
