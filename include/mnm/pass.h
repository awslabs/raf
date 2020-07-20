/*!
 * Copyright (c) 2019 by Contributors
 * \file pass.h
 * \brief A compatibility layer for TVM/Relay passes
 */
#pragma once

#include "tvm/relay/analysis.h"
#include "mnm/ir.h"

namespace mnm {
namespace pass {
using tvm::relay::FreeVars;
using tvm::AsText;
ir::Function AutoDiff(ir::Function func);
ir::Expr FoldConstant(ir::Expr expr, ir::Module mod);
ir::Expr BindParam(ir::Function func, ir::Array<ir::Expr> args);
}  // namespace pass
}  // namespace mnm
