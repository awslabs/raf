/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/ty/utils.h
 * \brief Typing utils
 */
#pragma once
#include <tvm/relay/type.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/env_func.h>
#include <tvm/tir/expr.h>
#include "raf/op.h"
#include "../schema/transform.h"
#include "../schema/ufunc.h"

namespace raf {
namespace op {

/*! \brief Get the type of value.
 *
 * \param value The value whose type to be retrieved
 *
 * \return The type of value
 */
tvm::Type GetType(value::Value value);

/*! \brief Get the value in a DLTensor.
 *
 *  \param v BaseTensorValue which only contains one item
 *
 *  \return The value in Tensor
 */
template <typename T>
T GetScalarValue(value::BaseTensorValue v) {
  if (auto* tvo = v.as<value::TensorValueObj>()) {
    DLDevice cpu_ctx;
    cpu_ctx.device_type = kDLCPU;
    cpu_ctx.device_id = 0;
    tensor::Tensor tensor = tvo->tensor;
    CHECK(tensor->ndim == 0);
    tvm::runtime::NDArray cpu_array = tensor.CopyTo(cpu_ctx);
    return reinterpret_cast<T*>(cpu_array->data)[0];
  }
  LOG(FATAL) << "Cannot convert " << v->GetTypeKey() << " to scalar";
}

/*! \brief Calculate the output size of arange OP.
 *
 *  \param ArangeArgs, arange arguments
 *
 *  \return The size of output
 */
template <typename T>
int32_t CalArangeOutputSize(const schema::ArangeArgs* args) {
  T start_v = GetScalarValue<T>(args->start);
  T stop_v = GetScalarValue<T>(args->stop);
  T step_v = GetScalarValue<T>(args->step);
  return (int32_t)ceil(((double)stop_v - start_v) / step_v);
}

/*! \brief Checks if the two PrimExpr matches the comparator.
 * If the result cannot be determined, it is considered true.
 * If inputs contain Any, it is considered true.
 * This is useful for expr containing variables.
 *
 * \param lhs PrimExpr on the left hand side
 * \param rhs PrimExpr on the right hand side
 * \param compare Comparator between lhs and rhs
 *
 * \return Whether the two PrimExpr matches the comparator.
 */
template <typename Comparator>
bool TypeCheckCompare(const tvm::PrimExpr& lhs, const tvm::PrimExpr& rhs, Comparator compare) {
  if (lhs.as<tvm::tir::AnyNode>() || rhs.as<tvm::tir::AnyNode>()) {
    return true;
  }
  tvm::PrimExpr diff = lhs - rhs;
  if (const int64_t* pdiff = tvm::tir::as_const_int(diff)) {
    return compare(pdiff[0], 0);
  }
  return true;
}

/*! \brief Checks if a PrimExpr evaluates to true.
 * If the result cannot be determined, it is considered true.
 * This is useful for expr containing variables.
 *
 * \param cond The condition to be evaluated
 *
 * \return The evaluated boolean value
 */
bool TypeCheck(const tvm::PrimExpr& cond);

template <typename T>
tvm::Type GeneralDxInfer(const CallValues& value) {
  const auto* args = value->args.as<T>();
  CHECK(args != nullptr);
  tvm::TensorType x = tvm::Downcast<tvm::TensorType>(GetType(args->x));
  return x;
}

inline Array<PrimExpr> BroadcastShape(const TensorType& x1, const TensorType& x2) {
  size_t ndim_1 = x1->shape.size();
  size_t ndim_2 = x2->shape.size();
  size_t ndim = std::max(ndim_1, ndim_2);
  Array<PrimExpr> oshape(ndim, 0);
  for (size_t i = 0; i < ndim; ++i) {
    PrimExpr lhs = (i < ndim_1) ? x1->shape[ndim_1 - 1 - i] : Integer(1);
    PrimExpr rhs = (i < ndim_2) ? x2->shape[ndim_2 - 1 - i] : Integer(1);

    if (tvm::tir::is_const_int(lhs, 1)) {
      oshape.Set(ndim - 1 - i, rhs);
    } else if (tvm::tir::is_const_int(rhs, 1)) {
      oshape.Set(ndim - 1 - i, lhs);
    } else if (lhs.as<AnyNode>()) {
      oshape.Set(ndim - 1 - i, rhs);
    } else if (rhs.as<AnyNode>()) {
      oshape.Set(ndim - 1 - i, lhs);
    } else if (TypeCheckCompare(lhs, rhs, std::equal_to<int>())) {
      oshape.Set(ndim - 1 - i, lhs);
    } else {
      LOG(FATAL) << "Incompatible broadcast type " << x1 << " and " << x2;
    }
  }
  return oshape;
}

}  // namespace op
}  // namespace raf
