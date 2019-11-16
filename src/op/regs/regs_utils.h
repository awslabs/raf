/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/regs/regs_utils.h
 * \brief Helpers for operator registry
 */
#pragma once
#include <vector>
#include <string>
#include "mnm/op.h"
#include "mnm/ir.h"
#include "mnm/registry.h"
#include "mnm/value.h"

namespace mnm {
namespace op {
namespace ffi {
ir::Expr ToAny(const registry::TVMArgValue& a);
ir::Expr ToTensor(const registry::TVMArgValue& a);
ir::Expr ToInt(const registry::TVMArgValue& a);
ir::Expr ToBool(const registry::TVMArgValue& a);
ir::Expr ToDouble(const registry::TVMArgValue& a);
ir::Expr ToString(const registry::TVMArgValue& a);
ir::Expr ToIntTuple(const registry::TVMArgValue& a);
ir::Expr ToOptionalIntTuple(const registry::TVMArgValue& a);
}  // namespace ffi
}  // namespace op
}  // namespace mnm

namespace mnm {
namespace op {
namespace args {
value::Value ToAny(const value::Value& a);
value::TensorValue ToTensor(const value::Value& a);
int64_t ToInt(const value::Value& a);
bool ToBool(const value::Value& a);
double ToDouble(const value::Value& a);
std::string ToString(const value::Value& a);
std::vector<int64_t> ToIntTuple(const value::Value& a);
std::vector<int64_t> ToOptionalIntTuple(const value::Value& a);
}  // namespace args
}  // namespace op
}  // namespace mnm
