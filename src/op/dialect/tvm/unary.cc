/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file ./src/op/dialect/tvm/unary.cc
 * \brief Unary operators bridged from TVM.
 */
#include <array>
#include "./tvm_attrs.h"
#include "./tvm_utils.h"
#include "../../schema/ufunc.h"
#include "../../../common/shape_utils.h"

namespace raf {
namespace op {
namespace tvm_dialect {

using namespace raf::ir;
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

#define RAF_TVM_UNARY(OP, FUNC)                                                                    \
  RAF_TVM(OP, FUNC, UnaryArgs, UnarySchema2Args, UnarySchemaArgNames, GenericAttrs, GenericHasher, \
          kElemWise)

#define RAF_TVM_UNARY_DX(OP, FUNC)                                                               \
  RAF_TVM(OP, FUNC, UnaryDxArgs, UnaryDxSchema2Args, UnaryDxSchemaArgNames, UnaryDxSchema2Attrs, \
          GenericHasher, kElemWise)

RAF_TVM_UNARY(copy, Copy);
RAF_TVM_UNARY(abs, Abs);
RAF_TVM_UNARY(ceil, Ceil);
RAF_TVM_UNARY(floor, Floor);
RAF_TVM_UNARY(log, Log);
RAF_TVM_UNARY(log2, Log2);
RAF_TVM_UNARY(exp, Exp);
RAF_TVM_UNARY(cos, Cos);
RAF_TVM_UNARY(sin, Sin);
RAF_TVM_UNARY(sign, Sign);
RAF_TVM_UNARY(round, Round);
RAF_TVM_UNARY(relu, Relu);
RAF_TVM_UNARY(gelu, Gelu);
RAF_TVM_UNARY(erf, Erf);
RAF_TVM_UNARY(sqrt, Sqrt);
RAF_TVM_UNARY(rsqrt, Rsqrt);
RAF_TVM_UNARY(atan, Atan);
RAF_TVM_UNARY(negative, Negative);
RAF_TVM_UNARY(sigmoid, Sigmoid);
RAF_TVM_UNARY(tanh, Tanh);
RAF_TVM_UNARY(batch_flatten, BatchFlatten);
RAF_TVM_UNARY(zeros_like, ZerosLike);
RAF_TVM_UNARY(ones_like, OnesLike);
RAF_TVM_UNARY(trunc, Trunc);

RAF_TVM_UNARY_DX(gelu_dx, GeluDx);
RAF_TVM_UNARY_DX(erf_dx, ErfDx);
RAF_TVM_UNARY_DX(sqrt_dx, SqrtDx);
RAF_TVM_UNARY_DX(tanh_dx, TanhDx);
RAF_TVM_PLEVEL(relu_dx, ReluDx, UnaryDxArgs, UnaryDxSchema2Args, UnaryDxSchemaArgNames,
               UnaryDxSchema2Attrs, GenericHasher, kElemWise, 20);

}  // namespace tvm_dialect
}  // namespace op
}  // namespace raf
