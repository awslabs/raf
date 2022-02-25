/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/declare/unary.cc
 * \brief Declaration of unary operators
 */
#include "raf/op.h"
#include "raf/op_utils.h"
#include "raf/tensor.h"
#include "../schema/init.h"

namespace raf {
namespace op {
namespace init {

using namespace raf::op::schema;
using namespace raf::value;

RAF_OP_DECLARE("raf.op.zeros", [](const CallValues& call) {
  const auto* args = call->args.as<InitOpArgs>();
  CHECK(args != nullptr);
  std::string dtype = args->dtype;
  std::vector<int64_t> shape = GetShapeVecFromValue(args->shape);
  const auto* f = tvm::runtime::Registry::Get("raf._core.core_utils.str2dev");
  tvm::Device tvm_dev = (*f)(args->device);
  Device device(tvm_dev);
  call->out = TensorValue::Assemble(/*dev=*/device,
                                    /*dtype=*/ir::String2DLDataType(dtype),
                                    /*shape=*/shape);
  call->device = device;
});

RAF_OP_DECLARE("raf.op.ones", [](const CallValues& call) {
  const auto* args = call->args.as<InitOpArgs>();
  CHECK(args != nullptr);
  std::string dtype = args->dtype;
  std::vector<int64_t> shape = GetShapeVecFromValue(args->shape);
  const auto* f = tvm::runtime::Registry::Get("raf._core.core_utils.str2dev");
  tvm::Device tvm_dev = (*f)(args->device);
  Device device(tvm_dev);
  call->out = TensorValue::Assemble(/*dev=*/device,
                                    /*dtype=*/ir::String2DLDataType(dtype),
                                    /*shape=*/shape);
  call->device = device;
});

RAF_OP_DECLARE("raf.op.one_hot", [](const CallValues& call) {
  const auto* args = call->args.as<OneHotArgs>();
  CHECK(args != nullptr);
  std::string dtype = args->dtype;
  DLTensor* indices = args->indices;
  std::vector<int64_t> shape(indices->shape, indices->shape + indices->ndim);
  CHECK_GE(args->depth, 0);
  shape.push_back(args->depth);
  const auto* f = tvm::runtime::Registry::Get("raf._core.core_utils.str2dev");
  tvm::Device tvm_dev = (*f)(args->device);
  Device device(tvm_dev);
  call->out = TensorValue::Assemble(/*dev=*/device,
                                    /*dtype=*/ir::String2DLDataType(dtype),
                                    /*shape=*/shape);
  call->device = device;
});

}  // namespace init
}  // namespace op
}  // namespace raf
