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
 * \file ./src/op/dialect/tvm/unary.cc
 * \brief Unary operators bridged from TVM.
 */
#include <array>
#include "./tvm_attrs.h"
#include "./tvm_utils.h"
#include "../../schema/ufunc.h"
#include "../../../common/shape_utils.h"

namespace mnm {
namespace op {
namespace tvm_dialect {

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

#define MNM_TVM_UNARY(OP, FUNC)                                                                    \
  MNM_TVM(OP, FUNC, UnaryArgs, UnarySchema2Args, UnarySchemaArgNames, GenericAttrs, GenericHasher, \
          kElemWise)

#define MNM_TVM_UNARY_DX(OP, FUNC)                                                               \
  MNM_TVM(OP, FUNC, UnaryDxArgs, UnaryDxSchema2Args, UnaryDxSchemaArgNames, UnaryDxSchema2Attrs, \
          GenericHasher, kElemWise)

MNM_TVM_UNARY(copy, Copy);
MNM_TVM_UNARY(abs, Abs);
MNM_TVM_UNARY(ceil, Ceil);
MNM_TVM_UNARY(floor, Floor);
MNM_TVM_UNARY(log, Log);
MNM_TVM_UNARY(log2, Log2);
MNM_TVM_UNARY(exp, Exp);
MNM_TVM_UNARY(cos, Cos);
MNM_TVM_UNARY(sin, Sin);
MNM_TVM_UNARY(sign, Sign);
MNM_TVM_UNARY(round, Round);
MNM_TVM_UNARY(relu, Relu);
MNM_TVM_UNARY(gelu, Gelu);
MNM_TVM_UNARY(erf, Erf);
MNM_TVM_UNARY(sqrt, Sqrt);
MNM_TVM_UNARY(rsqrt, Rsqrt);
MNM_TVM_UNARY(atan, Atan);
MNM_TVM_UNARY(negative, Negative);
MNM_TVM_UNARY(sigmoid, Sigmoid);
MNM_TVM_UNARY(tanh, Tanh);
MNM_TVM_UNARY(batch_flatten, BatchFlatten);
MNM_TVM_UNARY(zeros_like, ZerosLike);
MNM_TVM_UNARY(ones_like, OnesLike);
MNM_TVM_UNARY(trunc, Trunc);

MNM_TVM_UNARY_DX(gelu_dx, GeluDx);
MNM_TVM_UNARY_DX(erf_dx, ErfDx);
MNM_TVM_UNARY_DX(sqrt_dx, SqrtDx);
MNM_TVM_UNARY_DX(tanh_dx, TanhDx);
MNM_TVM_PLEVEL(relu_dx, ReluDx, UnaryDxArgs, UnaryDxSchema2Args, UnaryDxSchemaArgNames,
               UnaryDxSchema2Attrs, GenericHasher, kElemWise, 20);

}  // namespace tvm_dialect
}  // namespace op
}  // namespace mnm
