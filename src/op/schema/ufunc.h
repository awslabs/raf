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
 * \file src/op/schema/ufunc.h
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
class BinaryArgs : public ir::AttrsNode<BinaryArgs> {
 public:
  value::Value x1;
  value::Value x2;
  MNM_OP_SCHEMA(BinaryArgs, "mnm.args.binary");
};

class BinaryDxArgs : public ir::AttrsNode<BinaryDxArgs> {
 public:
  value::Value x1;
  value::Value x2;
  value::BaseTensorValue y;
  value::BaseTensorValue dy;
  MNM_OP_SCHEMA(BinaryDxArgs, "mnm.args.binary_dx");
};

class BinaryUfuncArgs : public ir::AttrsNode<BinaryUfuncArgs> {
 public:
  value::Value x1;
  value::Value x2;
  value::Value out{nullptr};
  value::Value where{nullptr};
  MNM_OP_SCHEMA(BinaryUfuncArgs, "mnm.args.binary_ufunc");
};

class TernaryArgs : public ir::AttrsNode<TernaryArgs> {
 public:
  value::Value x1;
  value::Value x2;
  value::Value x3;
  MNM_OP_SCHEMA(TernaryArgs, "mnm.args.ternary");
};

class TernaryDxArgs : public ir::AttrsNode<TernaryDxArgs> {
 public:
  value::Value x1;
  value::Value x2;
  value::Value x3;
  value::BaseTensorValue y;
  value::BaseTensorValue dy;
  MNM_OP_SCHEMA(TernaryDxArgs, "mnm.args.ternary_dx");
};

class TernaryUfuncArgs : public ir::AttrsNode<TernaryUfuncArgs> {
 public:
  value::Value x1;
  value::Value x2;
  value::Value x3;
  value::Value out{nullptr};
  value::Value where{nullptr};
  MNM_OP_SCHEMA(TernaryUfuncArgs, "mnm.args.ternary_ufunc");
};

class UnaryArgs : public ir::AttrsNode<UnaryArgs> {
 public:
  value::Value x;
  MNM_OP_SCHEMA(UnaryArgs, "mnm.args.unary");
};

class UnaryDxArgs : public ir::AttrsNode<UnaryDxArgs> {
 public:
  ir::Optional<value::Value> x;
  ir::Optional<value::BaseTensorValue> y;
  value::BaseTensorValue dy;
  MNM_OP_SCHEMA(UnaryDxArgs, "mnm.args.unary_dx");
};

class UnaryUfuncArgs : public ir::AttrsNode<UnaryUfuncArgs> {
 public:
  value::Value x;
  value::Value out{nullptr};
  value::Value where{nullptr};
  MNM_OP_SCHEMA(UnaryUfuncArgs, "mnm.args.unary_ufunc");
};
}  // namespace schema
}  // namespace op
}  // namespace mnm
