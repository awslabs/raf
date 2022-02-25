/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/ty/annotation.cc
 * \brief Typing of annotation operators
 */
#include <tvm/relay/type.h>
#include "raf/type.h"
#include "../schema/ufunc.h"
#include "./utils.h"

namespace raf {
namespace op {

using namespace raf::ir;
using namespace schema;

Type CompilerInfer(const CallValues& value) {
  const auto* args = value->args.as<UnaryArgs>();
  CHECK(args != nullptr);
  return GetType(args->x);
}

RAF_OP_TYPE("raf.op.compiler_begin", "Compiler", CompilerInfer);
RAF_OP_TYPE("raf.op.compiler_end", "Compiler", CompilerInfer);

}  // namespace op
}  // namespace raf
