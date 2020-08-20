/*!
 * Copyright (c) 2019 by Contributors
 * \file pass.h
 * \brief A compatibility layer for TVM/Relay passes
 */
#pragma once

#include "tvm/relay/analysis.h"
#include "tvm/relay/transform.h"
#include "mnm/ir.h"
#include "mnm/ir_ext.h"

namespace mnm {
namespace pass {
using tvm::AsText;
using tvm::relay::FreeVars;
ir::Function AutoDiff(ir::Function func);
ir::Expr FoldConstant(ir::Expr expr, ir::Module mod);
ir::Expr BindParam(ir::Function func, ir::Array<ir::Expr> args);
ir::Module LambdaLift(ir::Module mod);
/*!
 * \brief Manifest memory allocation.
 * \param mod The IR module.
 * \return Transformed IR module.
 */
ir::Module ManifestAlloc(ir::Module mod);
ir::Expr CanonicalizeOps(ir::Expr expr);
}  // namespace pass
}  // namespace mnm
