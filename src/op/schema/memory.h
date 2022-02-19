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
