/*!
 * Copyright (c) 2019 by Contributors
 * \file ./src/op/dispatch/tvmjit/binary.cc
 * \brief Binary operators bridged from TVM.
 */
#include <array>
#include <mnm/tvmjit/reduce.h>
#include "./tvm_attrs.h"
#include "./tvmjit_utils.h"
#include "../../schema/likes.h"
#include "../../../common/shape_utils.h"

namespace mnm {
namespace op {
namespace tvmjit {

using namespace mnm::ir;
using namespace mnm::op::schema;

MNM_TVMJIT(Add, "mnm.op.add", BinaryUfuncArgs, BinarySchema2Args, BinarySchemaArgNames,
           GenericAttrs, GenericHasher);
MNM_TVMJIT(Subtract, "mnm.op.subtract", BinaryUfuncArgs, BinarySchema2Args, BinarySchemaArgNames,
           GenericAttrs, GenericHasher);
MNM_TVMJIT(Divide, "mnm.op.divide", BinaryArgs, BinarySchema2Args, BinarySchemaArgNames,
           GenericAttrs, GenericHasher);
MNM_TVMJIT(Multiply, "mnm.op.multiply", BinaryArgs, BinarySchema2Args, BinarySchemaArgNames,
           GenericAttrs, GenericHasher);
MNM_TVMJIT(Power, "mnm.op.power", BinaryArgs, BinarySchema2Args, BinarySchemaArgNames, GenericAttrs,
           GenericHasher);
MNM_TVMJIT(Greater, "mnm.op.greater", BinaryArgs, BinarySchema2Args, BinarySchemaArgNames,
           GenericAttrs, GenericHasher);
MNM_TVMJIT(Maximum, "mnm.op.maximum", BinaryArgs, BinarySchema2Args, BinarySchemaArgNames,
           GenericAttrs, GenericHasher);
MNM_TVMJIT(Minimum, "mnm.op.minimum", BinaryArgs, BinarySchema2Args, BinarySchemaArgNames,
           GenericAttrs, GenericHasher);
MNM_TVMJIT(LogicalAnd, "mnm.op.logical_and", BinaryArgs, BinarySchema2Args, BinarySchemaArgNames,
           GenericAttrs, GenericHasher);
MNM_TVMJIT(Right_shift, "mnm.op.right_shift", BinaryArgs, BinarySchema2Args, BinarySchemaArgNames,
           GenericAttrs, GenericHasher);
MNM_TVMJIT(LeftShift, "mnm.op.left_shift", BinaryArgs, BinarySchema2Args, BinarySchemaArgNames,
           GenericAttrs, GenericHasher);
MNM_TVMJIT(NotEqual, "mnm.op.not_equal", BinaryArgs, BinarySchema2Args, BinarySchemaArgNames,
           GenericAttrs, GenericHasher);
}  // namespace tvmjit
}  // namespace op
}  // namespace mnm
