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
 * \file src/op/schema/init.h
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
class InitOpArgs : public ir::AttrsNode<InitOpArgs> {
 public:
  value::Value shape;
  std::string dtype{"int"};
  std::string device{"cpu"};
  MNM_OP_SCHEMA(InitOpArgs, "mnm.args.init_op");
};

class OneHotArgs : public ir::AttrsNode<OneHotArgs> {
 public:
  value::BaseTensorValue indices;
  value::BaseTensorValue on_value;
  value::BaseTensorValue off_value;
  int64_t depth;
  int64_t axis{-1};
  std::string dtype{"int"};
  std::string device{"cpu"};
  MNM_OP_SCHEMA(OneHotArgs, "mnm.args.one_hot");
};
}  // namespace schema
}  // namespace op
}  // namespace mnm
