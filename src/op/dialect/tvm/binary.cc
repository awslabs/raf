/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
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

namespace raf {
namespace op {
namespace tvm_dialect {

using namespace raf::ir;
using namespace raf::op::schema;

#define RAF_TVM_BINARY(OP, FUNC, SCHEMA)                                                          \
  RAF_TVM(OP, FUNC, SCHEMA, BinarySchema2Args, BinarySchemaArgNames, GenericAttrs, GenericHasher, \
          kBroadcast)

RAF_TVM_BINARY(add, Add, BinaryUfuncArgs);
RAF_TVM_BINARY(subtract, Subtract, BinaryUfuncArgs);
RAF_TVM_BINARY(divide, Divide, BinaryArgs);
RAF_TVM_BINARY(floor_divide, FloorDivide, BinaryArgs);
RAF_TVM_BINARY(multiply, Multiply, BinaryArgs);
RAF_TVM_BINARY(power, Power, BinaryArgs);
RAF_TVM_BINARY(maximum, Maximum, BinaryArgs);
RAF_TVM_BINARY(minimum, Minimum, BinaryArgs);
RAF_TVM_BINARY(logical_and, LogicalAnd, BinaryArgs);
RAF_TVM_BINARY(right_shift, Right_shift, BinaryArgs);
RAF_TVM_BINARY(left_shift, LeftShift, BinaryArgs);
RAF_TVM_BINARY(equal, Equal, BinaryArgs);
RAF_TVM_BINARY(not_equal, NotEqual, BinaryArgs);
RAF_TVM_BINARY(less, Less, BinaryArgs);
RAF_TVM_BINARY(less_equal, LessEqual, BinaryArgs);
RAF_TVM_BINARY(greater, Greater, BinaryArgs);
RAF_TVM_BINARY(greater_equal, GreaterEqual, BinaryArgs);

}  // namespace tvm_dialect
}  // namespace op
}  // namespace raf
