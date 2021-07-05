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

class RecvArgs : public ir::AttrsNode<RecvArgs> {
 public:
  int peer;
  std::vector<int64_t> shape;
  std::string dtype{"float32"};
  MNM_OP_SCHEMA(RecvArgs, "mnm.args.recv");
};

class ReduceScatterArgs : public ir::AttrsNode<ReduceScatterArgs> {
 public:
  std::vector<value::BaseTensorValue> x;
  MNM_OP_SCHEMA(ReduceScatterArgs, "mnm.args.reduce_scatter");
};

class SendArgs : public ir::AttrsNode<SendArgs> {
 public:
  value::BaseTensorValue x;
  int peer;
  MNM_OP_SCHEMA(SendArgs, "mnm.args.send");
};
}  // namespace schema
}  // namespace op
}  // namespace mnm
