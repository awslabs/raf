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
 * \file src/op/schema/likes.h
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
class CollapseLikeArgs : public ir::AttrsNode<CollapseLikeArgs> {
 public:
  value::BaseTensorValue x;
  std::vector<int64_t> shape;
  MNM_OP_SCHEMA(CollapseLikeArgs, "mnm.args.collapse_like");
};

class ReshapeArgs : public ir::AttrsNode<ReshapeArgs> {
 public:
  value::BaseTensorValue x;
  value::Value shape;
  bool reverse{false};
  MNM_OP_SCHEMA(ReshapeArgs, "mnm.args.reshape");
};

class Resize2DArgs : public ir::AttrsNode<Resize2DArgs> {
 public:
  value::BaseTensorValue x;
  value::Value size;
  std::string layout{"NCHW"};
  std::string method{"linear"};
  std::string coordinate_transformation_mode{"half_pixel"};
  std::string rounding_method{};
  float cubic_alpha{-0.5};
  int cubic_exclude{0};
  std::string out_dtype{};
  MNM_OP_SCHEMA(Resize2DArgs, "mnm.args.resize2d");
};

class Resize2DDxArgs : public ir::AttrsNode<Resize2DDxArgs> {
 public:
  value::BaseTensorValue x;
  value::BaseTensorValue dy;
  std::vector<int64_t> size;
  std::string layout{"NCHW"};
  std::string method{"linear"};
  std::string coordinate_transformation_mode{"half_pixel"};
  std::string rounding_method{};
  float cubic_alpha{-0.5};
  int cubic_exclude{0};
  std::string out_dtype{};
  MNM_OP_SCHEMA(Resize2DDxArgs, "mnm.args.resize2d_dx");
};

class SumArgs : public ir::AttrsNode<SumArgs> {
 public:
  value::BaseTensorValue x;
  std::vector<int64_t> axis{};
  std::vector<int64_t> keepdims{0};
  bool exclude{false};
  MNM_OP_SCHEMA(SumArgs, "mnm.args.sum");
};

class SumDxArgs : public ir::AttrsNode<SumDxArgs> {
 public:
  value::BaseTensorValue x;
  value::BaseTensorValue dy;
  std::vector<int64_t> axis{};
  std::vector<int64_t> keepdims{0};
  bool exclude{false};
  MNM_OP_SCHEMA(SumDxArgs, "mnm.args.sum_dx");
};
}  // namespace schema
}  // namespace op
}  // namespace mnm
