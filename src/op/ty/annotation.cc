/*!
 * Copyright (c) 2021 by Contributors
 * \file src/op/ty/annotation.cc
 * \brief Typing of annotation operators
 */
#include <tvm/relay/type.h>
#include "mnm/type.h"
#include "../schema/annotation.h"
#include "./utils.h"

namespace mnm {
namespace op {
namespace type {

using namespace schema;
using tvm::relay::Type;

Type CompilerInfer(const CallValues& value) {
  const auto* args = value->args.as<CompilerArgs>();
  CHECK(args != nullptr);
  return GetType(args->x);
}

MNM_OP_TYPE("mnm.op.compiler_begin", "Compiler", CompilerInfer);
MNM_OP_TYPE("mnm.op.compiler_end", "Compiler", CompilerInfer);

}  // namespace type
}  // namespace op
}  // namespace mnm
