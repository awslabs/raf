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
class AllgatherArgs : public ir::AttrsNode<AllgatherArgs> {
 public:
  value::BaseTensorValue x;
  int axis;
  MNM_OP_SCHEMA(AllgatherArgs, "mnm.args.allgather");
};

class AllreduceArgs : public ir::AttrsNode<AllreduceArgs> {
 public:
  std::vector<value::BaseTensorValue> x;
  std::string computation{"sum"};
  MNM_OP_SCHEMA(AllreduceArgs, "mnm.args.allreduce");
};

class BroadcastArgs : public ir::AttrsNode<BroadcastArgs> {
 public:
  std::vector<value::BaseTensorValue> x;
  int root;
  MNM_OP_SCHEMA(BroadcastArgs, "mnm.args.broadcast");
};

class CommReduceArgs : public ir::AttrsNode<CommReduceArgs> {
 public:
  std::vector<value::BaseTensorValue> x;
  int root;
  std::string computation{"sum"};
  MNM_OP_SCHEMA(CommReduceArgs, "mnm.args.comm_reduce");
};

class RecvArgs : public ir::AttrsNode<RecvArgs> {
 public:
  int peer;
  std::vector<int64_t> shape;
  std::string dtype{"float32"};
  ir::Optional<value::BaseTensorValue> token{nullptr};
  MNM_OP_SCHEMA(RecvArgs, "mnm.args.recv");
};

class ReduceScatterArgs : public ir::AttrsNode<ReduceScatterArgs> {
 public:
  std::vector<value::BaseTensorValue> x;
  std::string computation{"sum"};
  MNM_OP_SCHEMA(ReduceScatterArgs, "mnm.args.reduce_scatter");
};

class SendArgs : public ir::AttrsNode<SendArgs> {
 public:
  value::BaseTensorValue x;
  int peer;
  ir::Optional<value::BaseTensorValue> token{nullptr};
  MNM_OP_SCHEMA(SendArgs, "mnm.args.send");
};
}  // namespace schema
}  // namespace op
}  // namespace mnm
