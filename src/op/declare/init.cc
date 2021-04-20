/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/declare/unary.cc
 * \brief Declaration of unary operators
 */
#include "mnm/op.h"
#include "mnm/tensor.h"
#include "../schema/init.h"

namespace mnm {
namespace op {
namespace init {

using namespace mnm::op::schema;
using namespace mnm::value;

MNM_OP_DECLARE("mnm.op.zeros", [](const CallValues& call) {
  const auto* args = call->args.as<InitOpArgs>();
  CHECK(args != nullptr);
  std::string dtype = args->dtype;
  std::vector<int64_t> shape(args->shape);
  const auto* f = tvm::runtime::Registry::Get("mnm._core.core_utils.str2dev");
  tvm::Device tvm_dev = (*f)(args->device);
  Device device(tvm_dev);
  call->out = TensorValue::Assemble(/*dev=*/device,
                                    /*dtype=*/ir::String2DLDataType(dtype),
                                    /*shape=*/shape);
  call->device = device;
}).set_attr<TOpPattern>("TOpPattern", kElemWise);

MNM_OP_DECLARE("mnm.op.ones", [](const CallValues& call) {
  const auto* args = call->args.as<InitOpArgs>();
  CHECK(args != nullptr);
  std::string dtype = args->dtype;
  std::vector<int64_t> shape(args->shape);
  const auto* f = tvm::runtime::Registry::Get("mnm._core.core_utils.str2dev");
  tvm::Device tvm_dev = (*f)(args->device);
  Device device(tvm_dev);
  call->out = TensorValue::Assemble(/*dev=*/device,
                                    /*dtype=*/ir::String2DLDataType(dtype),
                                    /*shape=*/shape);
  call->device = device;
}).set_attr<TOpPattern>("TOpPattern", kElemWise);

MNM_OP_DECLARE("mnm.op.one_hot", [](const CallValues& call) {
  const auto* args = call->args.as<OneHotArgs>();
  CHECK(args != nullptr);
  std::string dtype = args->dtype;
  DLTensor* indices = args->indices;
  std::vector<int64_t> shape(indices->shape, indices->shape + indices->ndim);
  CHECK_GE(args->depth, 0);
  shape.push_back(args->depth);
  const auto* f = tvm::runtime::Registry::Get("mnm._core.core_utils.str2dev");
  tvm::Device tvm_dev = (*f)(args->device);
  Device device(tvm_dev);
  call->out = TensorValue::Assemble(/*dev=*/device,
                                    /*dtype=*/ir::String2DLDataType(dtype),
                                    /*shape=*/shape);
  call->device = device;
}).set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable);

}  // namespace init
}  // namespace op
}  // namespace mnm
