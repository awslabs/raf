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
#include "../schema/nn.h"
#include "./utils.h"

namespace mnm {
namespace op {
namespace type {

using namespace mnm::value;
using schema::BiasAddArgs;
using tvm::relay::Type;

Type BiasAddInfer(const CallValues& value) {
  const auto* args = value->args.as<BiasAddArgs>();
  return GetType(args->x);
}

MNM_OP_TYPE("mnm.op.bias_add", "BiasAdd", BiasAddInfer);

}  // namespace type
}  // namespace op
}  // namespace mnm
