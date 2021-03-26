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

std::vector<std::string> UnarySchemaArgNames(const op::CallValues& call) {
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
MNM_TVMJIT(BatchFlatten, "mnm.op.batch_flatten", UnaryArgs, UnarySchema2Args, UnarySchemaArgNames,
           GenericAttrs, GenericHasher);
MNM_TVMJIT(ZerosLike, "mnm.op.zeros_like", UnaryArgs, UnarySchema2Args, UnarySchemaArgNames,
           GenericAttrs, GenericHasher);
MNM_TVMJIT(OnesLike, "mnm.op.ones_like", UnaryArgs, UnarySchema2Args, UnarySchemaArgNames,
           GenericAttrs, GenericHasher);
MNM_TVMJIT(Trunc, "mnm.op.trunc", UnaryArgs, UnarySchema2Args, UnarySchemaArgNames, GenericAttrs,
           GenericHasher);

struct UnaryDxAttr : public tvm::AttrsNode<UnaryDxAttr> {
  std::string grad_mode;

  TVM_DECLARE_ATTRS(UnaryDxAttr, "relay.attrs.UnaryDxAttr") {
    TVM_ATTR_FIELD(grad_mode).describe(
        "Indicate how to calculate the gradient: using input, output or both");
  }
};

TVM_REGISTER_NODE_TYPE(UnaryDxAttr);

std::vector<Value> UnaryDxSchema2Args(const UnaryDxArgs* args) {
  CHECK(args->x.defined() || args->y.defined());
  std::vector<Value> ret;
  if (args->x.defined()) {
    ret.push_back(args->x.value());
  }
  if (args->y.defined()) {
    ret.push_back(args->y.value());
  }
  ret.push_back(args->dy);
  return ret;
}

std::vector<std::string> UnaryDxSchemaArgNames(const op::CallValues& call) {
  const auto* args = call->args.as<UnaryDxArgs>();
  CHECK(args->x.defined() || args->y.defined());
  std::vector<std::string> ret;
  if (args->x.defined()) {
    ret.push_back("x");
  }
  if (args->y.defined()) {
    ret.push_back("y");
  }
  ret.push_back("dy");

  return ret;
}

Attrs UnaryDxSchema2Attrs(const UnaryDxArgs* args) {
  auto attrs = make_object<UnaryDxAttr>();
  CHECK(args->x.defined() || args->y.defined());
  attrs->grad_mode = "both";
  if (!args->x.defined()) {
    attrs->grad_mode = "output";
  } else if (!args->y.defined()) {
    attrs->grad_mode = "input";
  }
  return Attrs(attrs);
}

MNM_TVMJIT_PLEVEL(ReluDx, "mnm.op.relu_dx", UnaryDxArgs, UnaryDxSchema2Args, UnaryDxSchemaArgNames,
                  UnaryDxSchema2Attrs, GenericHasher, 20);
MNM_TVMJIT(ErfDx, "mnm.op.erf_dx", UnaryDxArgs, UnaryDxSchema2Args, UnaryDxSchemaArgNames,
           UnaryDxSchema2Attrs, GenericHasher);
MNM_TVMJIT(TanhDx, "mnm.op.tanh_dx", UnaryDxArgs, UnaryDxSchema2Args, UnaryDxSchemaArgNames,
           UnaryDxSchema2Attrs, GenericHasher);
}  // namespace tvmjit
}  // namespace op
}  // namespace mnm
