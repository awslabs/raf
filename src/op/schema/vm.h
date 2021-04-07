/*!
 * Copyright (c) 2020 by Contributors
 * Auto generated. Do not touch.
 * \file src/op/schema/vm.h
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
class AllocStorageArgs : public ir::AttrsNode<AllocStorageArgs> {
 public:
  value::Value size;
  value::Value alignment;
  int device_type;
  int device_id;
  std::string dtype{"float32"};
  MNM_OP_SCHEMA(AllocStorageArgs, "mnm.args.alloc_storage");
};

class AllocTensorArgs : public ir::AttrsNode<AllocTensorArgs> {
 public:
  value::BaseTensorValue storage;
  value::Value shape;
  std::string dtype{"float32"};
  std::vector<int64_t> assert_shape{};
  MNM_OP_SCHEMA(AllocTensorArgs, "mnm.args.alloc_tensor");
};

class InvokeOpArgs : public ir::AttrsNode<InvokeOpArgs> {
 public:
  value::Value func;
  value::Value inputs;
  value::Value outputs;
  MNM_OP_SCHEMA(InvokeOpArgs, "mnm.args.invoke_op");
};
}  // namespace schema
}  // namespace op
}  // namespace mnm
