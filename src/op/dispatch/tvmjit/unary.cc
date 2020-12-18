/*!
 * Copyright (c) 2019 by Contributors
 * \file ./src/op/dispatch/tvmjit/unary.cc
 * \brief Unary operators bridged from TVM.
 */
#include <array>
#include "./tvm_attrs.h"
#include "./tvmjit_utils.h"
#include "../../schema/ufunc.h"
#include "../../../common/shape_utils.h"

namespace mnm {
namespace op {
namespace tvmjit {

using namespace mnm::ir;
using common::shape_utils::GetNumel;
using schema::UnaryArgs;
using schema::UnaryDxArgs;
using schema::UnaryUfuncArgs;

std::vector<Value> UnarySchema2Args(const UnaryArgs* args) {
  return {args->x};
}

std::vector<std::string> UnarySchemaArgNames() {
  return {"x"};
}

MNM_TVMJIT(Copy, "mnm.op.copy", UnaryArgs, UnarySchema2Args, UnarySchemaArgNames, GenericAttrs,
           GenericHasher);
MNM_TVMJIT(Abs, "mnm.op.abs", UnaryArgs, UnarySchema2Args, UnarySchemaArgNames, GenericAttrs,
           GenericHasher);
MNM_TVMJIT(Ceil, "mnm.op.ceil", UnaryArgs, UnarySchema2Args, UnarySchemaArgNames, GenericAttrs,
           GenericHasher);
MNM_TVMJIT(Floor, "mnm.op.floor", UnaryArgs, UnarySchema2Args, UnarySchemaArgNames, GenericAttrs,
           GenericHasher);
MNM_TVMJIT(Log, "mnm.op.log", UnaryArgs, UnarySchema2Args, UnarySchemaArgNames, GenericAttrs,
           GenericHasher);
MNM_TVMJIT(Exp, "mnm.op.exp", UnaryArgs, UnarySchema2Args, UnarySchemaArgNames, GenericAttrs,
           GenericHasher);
MNM_TVMJIT(Cos, "mnm.op.cos", UnaryArgs, UnarySchema2Args, UnarySchemaArgNames, GenericAttrs,
           GenericHasher);
MNM_TVMJIT(Sin, "mnm.op.sin", UnaryArgs, UnarySchema2Args, UnarySchemaArgNames, GenericAttrs,
           GenericHasher);
MNM_TVMJIT(Sign, "mnm.op.sign", UnaryArgs, UnarySchema2Args, UnarySchemaArgNames, GenericAttrs,
           GenericHasher);
MNM_TVMJIT(Round, "mnm.op.round", UnaryArgs, UnarySchema2Args, UnarySchemaArgNames, GenericAttrs,
           GenericHasher);
MNM_TVMJIT(Relu, "mnm.op.relu", UnaryArgs, UnarySchema2Args, UnarySchemaArgNames, GenericAttrs,
           GenericHasher);
MNM_TVMJIT(Erf, "mnm.op.erf", UnaryArgs, UnarySchema2Args, UnarySchemaArgNames, GenericAttrs,
           GenericHasher);
MNM_TVMJIT(Sqrt, "mnm.op.sqrt", UnaryArgs, UnarySchema2Args, UnarySchemaArgNames, GenericAttrs,
           GenericHasher);
MNM_TVMJIT(Rsqrt, "mnm.op.rsqrt", UnaryArgs, UnarySchema2Args, UnarySchemaArgNames, GenericAttrs,
           GenericHasher);
MNM_TVMJIT(Atan, "mnm.op.atan", UnaryArgs, UnarySchema2Args, UnarySchemaArgNames, GenericAttrs,
           GenericHasher);
MNM_TVMJIT(Negative, "mnm.op.negative", UnaryArgs, UnarySchema2Args, UnarySchemaArgNames,
           GenericAttrs, GenericHasher);
MNM_TVMJIT(Sigmoid, "mnm.op.sigmoid", UnaryArgs, UnarySchema2Args, UnarySchemaArgNames,
           GenericAttrs, GenericHasher);
MNM_TVMJIT(Tanh, "mnm.op.tanh", UnaryArgs, UnarySchema2Args, UnarySchemaArgNames, GenericAttrs,
           GenericHasher);

std::vector<Value> UnaryDxSchema2Args(const UnaryDxArgs* args) {
  return {args->x, args->y, args->dy};
}

std::vector<std::string> UnaryDxSchemaArgNames() {
  return {"x", "y", "dy"};
}

MNM_TVMJIT(ReluDx, "mnm.op.relu_dx", UnaryDxArgs, UnaryDxSchema2Args, UnaryDxSchemaArgNames,
           GenericAttrs, GenericHasher);
MNM_TVMJIT(ErfDx, "mnm.op.erf_dx", UnaryDxArgs, UnaryDxSchema2Args, UnaryDxSchemaArgNames,
           GenericAttrs, GenericHasher);

}  // namespace tvmjit
}  // namespace op
}  // namespace mnm
