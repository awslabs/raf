/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
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
  int64_t stream_id{-1};
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
