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
 * \file src/op/schema/optimizer.h
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
class LansArgs : public ir::AttrsNode<LansArgs> {
 public:
  std::vector<value::BaseTensorValue> tensor_list;
  value::BaseTensorValue step;
  float learning_rate;
  float beta1;
  float beta2;
  float eps;
  int bias_correction;
  float weight_decay;
  int grad_averaging;
  int mode;
  bool normalize_grad;
  MNM_OP_SCHEMA(LansArgs, "mnm.args.lans");
};

class SgdArgs : public ir::AttrsNode<SgdArgs> {
 public:
  value::BaseTensorValue x;
  value::BaseTensorValue dx;
  value::BaseTensorValue v;
  double learning_rate;
  double mu;
  MNM_OP_SCHEMA(SgdArgs, "mnm.args.sgd");
};
}  // namespace schema
}  // namespace op
}  // namespace mnm
