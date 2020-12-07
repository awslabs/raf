/*!
 * Copyright (c) 2020 by Contributors
 * Auto generated. Do not touch.
 * \file src/op/schema/annotation.h
 * \brief Operator schema.
 */
#pragma once
#include <vector>
#include <string>
#include "mnm/op.h"
#include "mnm/value.h"
namespace mnm {
namespace op {
namespace schema {
class CompilerArgs : public ir::AttrsNode<CompilerArgs> {
 public:
  value::BaseTensorValue x;
  std::string compiler;
  MNM_OP_SCHEMA(CompilerArgs, "mnm.args.compiler");
};
}  // namespace schema
}  // namespace op
}  // namespace mnm
