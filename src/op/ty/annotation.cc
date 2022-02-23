/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/ty/annotation.cc
 * \brief Typing of annotation operators
 */
#include <tvm/relay/type.h>
#include "mnm/type.h"
#include "../schema/annotation.h"
#include "./utils.h"

namespace mnm {
namespace op {

using namespace mnm::ir;
using namespace schema;

Type CompilerInfer(const CallValues& value) {
  const auto* args = value->args.as<CompilerArgs>();
  CHECK(args != nullptr);
  return GetType(args->x);
}

MNM_OP_TYPE("mnm.op.compiler_begin", "Compiler", CompilerInfer);
MNM_OP_TYPE("mnm.op.compiler_end", "Compiler", CompilerInfer);

}  // namespace op
}  // namespace mnm
