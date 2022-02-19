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
 * \file src/op/schema/reduce.h
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
class L2NormArgs : public ir::AttrsNode<L2NormArgs> {
 public:
  value::BaseTensorValue x;
  MNM_OP_SCHEMA(L2NormArgs, "mnm.args.l2norm");
};

class MeanDxArgs : public ir::AttrsNode<MeanDxArgs> {
 public:
  value::BaseTensorValue dy;
  std::vector<int64_t> axis{};
  std::vector<int64_t> x_shape{};
  bool keepdims{false};
  bool exclude{false};
  MNM_OP_SCHEMA(MeanDxArgs, "mnm.args.mean_dx");
};

class ProdDxArgs : public ir::AttrsNode<ProdDxArgs> {
 public:
  value::BaseTensorValue x;
  value::BaseTensorValue dy;
  std::vector<int64_t> axis{};
  bool keepdims{false};
  bool exclude{false};
  MNM_OP_SCHEMA(ProdDxArgs, "mnm.args.prod_dx");
};

class ReduceArgs : public ir::AttrsNode<ReduceArgs> {
 public:
  value::BaseTensorValue x;
  std::vector<int64_t> axis{};
  bool keepdims{false};
  bool exclude{false};
  MNM_OP_SCHEMA(ReduceArgs, "mnm.args.reduce");
};
}  // namespace schema
}  // namespace op
}  // namespace mnm
