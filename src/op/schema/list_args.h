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
 * \file src/op/schema/list_args.h
 * \brief A list of arguments.
 */
#pragma once
#include <vector>
#include <string>
#include "mnm/op.h"

namespace mnm {
namespace op {
namespace schema {

class ListArgs : public ir::AttrsNode<ListArgs> {
 public:
  ir::Array<value::Value> args;
  MNM_OP_SCHEMA(ListArgs, "mnm.args.list");
};

}  // namespace schema
}  // namespace op
}  // namespace mnm
