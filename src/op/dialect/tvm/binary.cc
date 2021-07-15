/*!
 * Copyright (c) 2021 by Contributors
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
