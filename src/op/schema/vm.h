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
  bool own{true};
  MNM_OP_SCHEMA(AllocTensorArgs, "mnm.args.alloc_tensor");
};

class FreeArgs : public ir::AttrsNode<FreeArgs> {
 public:
  value::BaseTensorValue memory;
  MNM_OP_SCHEMA(FreeArgs, "mnm.args.free");
};

class InferTypeArgs : public ir::AttrsNode<InferTypeArgs> {
 public:
  value::Value func;
  value::Value inputs;
  MNM_OP_SCHEMA(InferTypeArgs, "mnm.args.infer_type");
};

class InvokeOpArgs : public ir::AttrsNode<InvokeOpArgs> {
 public:
  value::Value func;
  value::Value inputs;
  value::Value outputs;
  MNM_OP_SCHEMA(InvokeOpArgs, "mnm.args.invoke_op");
};

class SetShapeArgs : public ir::AttrsNode<SetShapeArgs> {
 public:
  value::BaseTensorValue data;
  value::Value shape;
  MNM_OP_SCHEMA(SetShapeArgs, "mnm.args.set_shape");
};
}  // namespace schema
}  // namespace op
}  // namespace mnm
