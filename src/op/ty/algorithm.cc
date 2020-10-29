/*!
 * Copyright (c) 2020 by Contributors
 * \file src/op/ty/algorithm.cc
 * \brief Typing of algorithm operators
 */
#include <tvm/relay/type.h>
#include "mnm/type.h"
#include "../schema/algorithm.h"
#include "./utils.h"

namespace mnm {
namespace op {
namespace type {

using namespace mnm::value;
using namespace schema;
using tvm::relay::Type;
using namespace tvm;
using namespace tvm::relay;

Type ArgsortInfer(const CallValues& value) {
  const auto* args = value->args.as<ArgsortArgs>();
  CHECK(args != nullptr);
  TensorType data = Downcast<TensorType>(GetType(args->data));
  DataType dtype = DataType(ir::String2DLDataType(args->dtype));
  return TensorType(data->shape, dtype);
}

MNM_OP_TYPE("mnm.op.argsort", "Argsort", ArgsortInfer);

}  // namespace type
}  // namespace op
}  // namespace mnm
