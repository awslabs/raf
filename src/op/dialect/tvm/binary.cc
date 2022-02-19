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
 * \file ./src/op/dialect/tvm_dialect/binary.cc
 * \brief Binary operators bridged from TVM.
 */
#include <array>
#include "./tvm_attrs.h"
#include "./tvm_utils.h"
#include "../../schema/ufunc.h"
#include "../../schema/likes.h"
#include "../../../common/shape_utils.h"

namespace mnm {
namespace op {
namespace tvm_dialect {

using namespace mnm::ir;
using namespace mnm::op::schema;

#define MNM_TVM_BINARY(OP, FUNC, SCHEMA)                                                          \
  MNM_TVM(OP, FUNC, SCHEMA, BinarySchema2Args, BinarySchemaArgNames, GenericAttrs, GenericHasher, \
          kBroadcast)

MNM_TVM_BINARY(add, Add, BinaryUfuncArgs);
MNM_TVM_BINARY(subtract, Subtract, BinaryUfuncArgs);
MNM_TVM_BINARY(divide, Divide, BinaryArgs);
MNM_TVM_BINARY(floor_divide, FloorDivide, BinaryArgs);
MNM_TVM_BINARY(multiply, Multiply, BinaryArgs);
MNM_TVM_BINARY(power, Power, BinaryArgs);
MNM_TVM_BINARY(maximum, Maximum, BinaryArgs);
MNM_TVM_BINARY(minimum, Minimum, BinaryArgs);
MNM_TVM_BINARY(logical_and, LogicalAnd, BinaryArgs);
MNM_TVM_BINARY(right_shift, Right_shift, BinaryArgs);
MNM_TVM_BINARY(left_shift, LeftShift, BinaryArgs);
MNM_TVM_BINARY(equal, Equal, BinaryArgs);
MNM_TVM_BINARY(not_equal, NotEqual, BinaryArgs);
MNM_TVM_BINARY(less, Less, BinaryArgs);
MNM_TVM_BINARY(less_equal, LessEqual, BinaryArgs);
MNM_TVM_BINARY(greater, Greater, BinaryArgs);
MNM_TVM_BINARY(greater_equal, GreaterEqual, BinaryArgs);

}  // namespace tvm_dialect
}  // namespace op
}  // namespace mnm
