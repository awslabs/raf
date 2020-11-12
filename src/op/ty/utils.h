/*!
 * Copyright (c) 2020 by Contributors
 * \file src/op/ty/utils.h
 * \brief Typing utils
 */
#pragma once
#include <tvm/relay/type.h>
#include <tvm/ir/attrs.h>
#include <tvm/node/container.h>
#include <tvm/ir/env_func.h>
#include <tvm/tir/expr.h>
#include "mnm/op.h"
#include "../schema/ufunc.h"

namespace mnm {
namespace op {
namespace type {

/*! \brief Get the type of value.
 *
 * \param value The value whose type to be retrieved
 *
 * \return The type of value
 */
tvm::Type GetType(value::Value value);

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
  using namespace tvm;
  if (lhs.as<tir::AnyNode>() || rhs.as<tir::AnyNode>()) {
    return true;
  }
  PrimExpr diff = lhs - rhs;
  if (const int64_t* pdiff = tir::as_const_int(diff)) {
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

}  // namespace type
}  // namespace op
}  // namespace mnm
