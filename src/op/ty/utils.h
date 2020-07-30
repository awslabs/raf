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

/*! \brief Checks if two PrimExpr are equal
 * If the result cannot be determined, they are considered equal.
 * If inputs contain Any, they are considered equal.
 * This is useful for expr containing variables.
 *
 * \param lhs PrimExpr on the left hand side
 * \param rhs PrimExpr on the right hand side
 *
 * \return Whether they are equal
 */
bool TypeCheckEqual(const tvm::PrimExpr& lhs, const tvm::PrimExpr& rhs);

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
