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
MNM_TVMJIT(Divide, "mnm.op.divide", BinaryUfuncArgs, BinarySchema2Args, BinarySchemaArgNames,
           GenericAttrs, GenericHasher);
MNM_TVMJIT(Multiply, "mnm.op.multiply", BinaryUfuncArgs, BinarySchema2Args, BinarySchemaArgNames,
           GenericAttrs, GenericHasher);
MNM_TVMJIT(Power, "mnm.op.power", BinaryUfuncArgs, BinarySchema2Args, BinarySchemaArgNames,
           GenericAttrs, GenericHasher);
MNM_TVMJIT(Greater, "mnm.op.greater", BinaryUfuncArgs, BinarySchema2Args, BinarySchemaArgNames,
           GenericAttrs, GenericHasher);
MNM_TVMJIT(Maximum, "mnm.op.maximum", BinaryUfuncArgs, BinarySchema2Args, BinarySchemaArgNames,
           GenericAttrs, GenericHasher);
MNM_TVMJIT(Minimum, "mnm.op.minimum", BinaryUfuncArgs, BinarySchema2Args, BinarySchemaArgNames,
           GenericAttrs, GenericHasher);
MNM_TVMJIT(LogicalAnd, "mnm.op.logical_and", BinaryUfuncArgs, BinarySchema2Args,
           BinarySchemaArgNames, GenericAttrs, GenericHasher);
MNM_TVMJIT(Right_shift, "mnm.op.right_shift", BinaryUfuncArgs, BinarySchema2Args,
           BinarySchemaArgNames, GenericAttrs, GenericHasher);
MNM_TVMJIT(LeftShift, "mnm.op.left_shift", BinaryUfuncArgs, BinarySchema2Args, BinarySchemaArgNames,
           GenericAttrs, GenericHasher);
MNM_TVMJIT(NotEqual, "mnm.op.not_equal", BinaryUfuncArgs, BinarySchema2Args, BinarySchemaArgNames,
           GenericAttrs, GenericHasher);
}  // namespace tvmjit
}  // namespace op
}  // namespace mnm
