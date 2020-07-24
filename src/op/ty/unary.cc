/*!
 * Copyright (c) 2020 by Contributors
 * \file src/op/ty/unary.cc
 * \brief Typing relations of unary operators
 */
#include <tvm/relay/type.h>
#include <tvm/ir/attrs.h>
#include <tvm/node/container.h>
#include <tvm/ir/env_func.h>
#include "mnm/type.h"
#include "../schema/ufunc.h"
#include "./utils.h"

namespace mnm {
namespace op {
namespace type {

using namespace mnm::value;
using tvm::relay::Type;
using schema::UnaryArgs;

Type UnaryInfer(const CallValues& value) {
  const auto* args = value->args.as<UnaryArgs>();
  // Unary ops' outputs are identical with inputs
  return GetType(args->x);
}

MNM_OP_TYPE("mnm.op.log", "Identity", UnaryInfer);
MNM_OP_TYPE("mnm.op.cos", "Identity", UnaryInfer);
MNM_OP_TYPE("mnm.op.relu", "Identity", UnaryInfer);

}  // namespace type
}  // namespace op
}  // namespace mnm
