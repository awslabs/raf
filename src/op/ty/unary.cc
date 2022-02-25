/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/ty/unary.cc
 * \brief Typing relations of unary operators
 */
#include <tvm/relay/type.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/env_func.h>
#include "raf/type.h"
#include "../schema/ufunc.h"
#include "./utils.h"

namespace raf {
namespace op {

using namespace raf::ir;
using namespace raf::value;
using schema::UnaryArgs;
using schema::UnaryDxArgs;
using schema::UnaryUfuncArgs;

Type UnaryInfer(const CallValues& value) {
  const auto* args = value->args.as<UnaryArgs>();
  CHECK(args != nullptr);
  // Unary ops' outputs are identical with inputs
  return GetType(args->x);
}

RAF_OP_TYPE("raf.op.log", "Identity", UnaryInfer);
RAF_OP_TYPE("raf.op.log2", "Identity", UnaryInfer);
RAF_OP_TYPE("raf.op.cos", "Identity", UnaryInfer);
RAF_OP_TYPE("raf.op.sin", "Identity", UnaryInfer);
RAF_OP_TYPE("raf.op.sign", "Identity", UnaryInfer);
RAF_OP_TYPE("raf.op.round", "Identity", UnaryInfer);
RAF_OP_TYPE("raf.op.relu", "Identity", UnaryInfer);
RAF_OP_TYPE("raf.op.gelu", "Identity", UnaryInfer);
RAF_OP_TYPE("raf.op.tanh", "Identity", UnaryInfer);
RAF_OP_TYPE("raf.op.sigmoid", "Identity", UnaryInfer);
RAF_OP_TYPE("raf.op.copy", "Identity", UnaryInfer);
RAF_OP_TYPE("raf.op.abs", "Identity", UnaryInfer);
RAF_OP_TYPE("raf.op.ceil", "Identity", UnaryInfer);
RAF_OP_TYPE("raf.op.floor", "Identity", UnaryInfer);
RAF_OP_TYPE("raf.op.exp", "Identity", UnaryInfer);
RAF_OP_TYPE("raf.op.erf", "Identity", UnaryInfer);
RAF_OP_TYPE("raf.op.sqrt", "Identity", UnaryInfer);
RAF_OP_TYPE("raf.op.rsqrt", "Identity", UnaryInfer);
RAF_OP_TYPE("raf.op.atan", "Identity", UnaryInfer);
RAF_OP_TYPE("raf.op.trunc", "Identity", UnaryInfer);
RAF_OP_TYPE("raf.op.ndarray_size", "Identity", UnaryInfer);

Type UnaryDxInfer(const CallValues& value) {
  const auto* args = value->args.as<UnaryDxArgs>();
  CHECK(args != nullptr);
  CHECK(args->x.defined() || args->y.defined());
  // Unary ops' outputs are identical with inputs
  if (args->x.defined()) {
    return GetType(args->x.value());
  } else {
    return GetType(args->y.value());
  }
}

RAF_OP_TYPE("raf.op.relu_dx", "IdentityDx", UnaryDxInfer);
RAF_OP_TYPE("raf.op.gelu_dx", "IdentityDx", UnaryDxInfer);
RAF_OP_TYPE("raf.op.tanh_dx", "IdentityDx", UnaryDxInfer);
RAF_OP_TYPE("raf.op.sigmoid_dx", "IdentityDx", UnaryDxInfer);
RAF_OP_TYPE("raf.op.erf_dx", "IdentityDx", UnaryDxInfer);
RAF_OP_TYPE("raf.op.sqrt_dx", "IdentityDx", UnaryDxInfer);

Type UnaryUfuncInfer(const CallValues& value) {
  const auto* args = value->args.as<UnaryUfuncArgs>();
  CHECK(args != nullptr);
  // UnaryUfunc ops' outputs are identical with inputs
  return GetType(args->x);
}

RAF_OP_TYPE("raf.op.negative", "Identity", UnaryInfer);
RAF_OP_TYPE("raf.op.logical_not", "Identity", UnaryInfer);

Type UnaryShapeInfer(const CallValues& value) {
  const auto* args = value->args.as<UnaryArgs>();
  TensorType x = Downcast<TensorType>(GetType(args->x));
  Array<tvm::PrimExpr> shape;
  shape.push_back(ir::Integer(x->shape.size()));
  return TensorType(shape, tvm::runtime::DataType::UInt(32));
}

RAF_OP_TYPE("raf.op.shape", "Shape", UnaryShapeInfer);
RAF_OP_TYPE("raf.op.zeros_like", "Identity", UnaryInfer);
RAF_OP_TYPE("raf.op.ones_like", "Identity", UnaryInfer);

Type NumelInfer(const CallValues& value) {
  const auto* args = value->args.as<UnaryArgs>();
  ICHECK(args != nullptr);
  ICHECK(args->x.defined());
  return TensorType({}, tvm::runtime::DataType::Int(32));
}

RAF_OP_TYPE("raf.op.numel", "Numel", NumelInfer);

Type ShapeAsTensorInfer(const CallValues& value) {
  const auto* args = value->args.as<UnaryArgs>();
  TensorType x = Downcast<TensorType>(GetType(args->x));
  Array<tvm::PrimExpr> shape;
  shape.push_back(ir::Integer(x->shape.size()));
  return TensorType(shape, tvm::runtime::DataType::Int(32));
}

RAF_OP_TYPE("raf.op.shape_as_tensor", "ShapeAsTensor", ShapeAsTensorInfer);

}  // namespace op
}  // namespace raf
