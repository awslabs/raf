/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/dialect/cutlass/cutlass_utils.cc
 * \brief Helper functions for cutlass
 */
#include <sstream>

#include "cutlass/library/singleton.h"
#include "raf/memory_pool.h"
#include "raf/value.h"

#include "./cutlass_utils.h"

namespace raf {
namespace op {
namespace cutlass {

using namespace raf::ir;
using namespace raf::value;

RAF_REGISTER_DIALECT("cutlass").set_enable(DevType::kCUDA());

CutlassOpEnv::CutlassOpEnv(const CallValues& call) : device_(call->device) {
  CUDA_CALL(cudaGetDeviceProperties(&device_prop_, device_.device_id()));
}

int CutlassOpEnv::compute_capability() const {
  return device_prop_.major * 10 + device_prop_.minor;
}

void CutlassOpEnv::Execute(const CallValues& call) {
  Array<Value> args = GetListArgs(call->args);
  std::vector<Value> inputs;
  for (const auto& i : arg_indices) {
    inputs.push_back(args[i]);
  }
  return Execute(inputs, call->out);
}

void CutlassOpEnv::RequestWorkspace(void** dest, const Device& device, int64_t nbytes) {
  workspace_mem_ = memory_pool::Memory::Alloc(device, nbytes);
  *dest = workspace_mem_->data;
}

std::ostream& operator<<(std::ostream& stream, const std::unique_ptr<TunableConfig>& config) {
  config->AsText(stream);
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const SplitKMode& mode) {
  if (mode == SplitKMode::kSerial) {
    stream << "kSerial";
  } else if (mode == SplitKMode::kParallel) {
    stream << "kParallel";
  } else {
    LOG(FATAL) << "Unknown SplitKMode";
  }
  return stream;
}

NumericTypeID GetNumericTypeID(DType dtype) {
  if (dtype.lanes != 1) {
    return NumericTypeID::kUnknown;
  }
  if (dtype.code == DTypeCode::kFloat()) {
    switch (dtype.bits) {
      case 16:
        return NumericTypeID::kF16;
      case 32:
        return NumericTypeID::kF32;
      case 64:
        return NumericTypeID::kF64;
    }
  }
  return NumericTypeID::kUnknown;
}

std::vector<int> GetArgIndices(const op::CallValues& call, const Array<Var>& params) {
  Function func = Downcast<ClosureValue>(call->callee)->func;
  std::vector<int> arg_indices;
  for (const auto& i : params) {
    size_t num = func->params.size();
    for (size_t j = 0; j < num; ++j) {
      if (i == func->params[j]) {
        arg_indices.push_back(j);
        break;
      }
    }
  }
  return arg_indices;
}

DType GetAccumulationDType(DType dtype) {
  if (dtype.lanes != 1) {
    return DType();
  }
  if (dtype.code == DTypeCode::kFloat()) {
    switch (dtype.bits) {
      case 16:
        return DType(dtype.code, 32);
      case 32:
      case 64:
        return dtype;
    }
  }
  return DType();
}

// Register auxilary ops
RAF_REGISTER_DIALECT_OP(cutlass, add, 0);
RAF_REGISTER_DIALECT_OP(cutlass, subtract, 0);
RAF_REGISTER_DIALECT_OP(cutlass, multiply, 0);
RAF_REGISTER_DIALECT_OP(cutlass, divide, 0);
RAF_REGISTER_DIALECT_OP(cutlass, relu, 0);
RAF_REGISTER_DIALECT_OP(cutlass, gelu, 0);

}  // namespace cutlass
}  // namespace op
}  // namespace raf
