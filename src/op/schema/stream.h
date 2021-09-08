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
class EventArgs : public ir::AttrsNode<EventArgs> {
 public:
  int64_t event_id;
  MNM_OP_SCHEMA(EventArgs, "mnm.args.event");
};

class SetStreamArgs : public ir::AttrsNode<SetStreamArgs> {
 public:
  int64_t device_id;
  int64_t stream_id;
  MNM_OP_SCHEMA(SetStreamArgs, "mnm.args.set_stream");
};

class StreamArgs : public ir::AttrsNode<StreamArgs> {
 public:
  value::BaseTensorValue x;
  int stream_tag{0};
  MNM_OP_SCHEMA(StreamArgs, "mnm.args.stream");
};

class StreamBarrierArgs : public ir::AttrsNode<StreamBarrierArgs> {
 public:
  MNM_OP_SCHEMA(StreamBarrierArgs, "mnm.args.stream_barrier");
};
}  // namespace schema
}  // namespace op
}  // namespace mnm
