/*!
 * Copyright (c) 2021 by Contributors
 * \file src/op/dispatch/cutlass/cutlass_utils.h
 * \brief Helper functions for cutlass
 */
#pragma once

#include "cutlass/library/library.h"
#include "cutlass/library/handle.h"
#include "cutlass_ext/library/operation_table_ext.h"
#include "cutlass_ext/library/singleton_ext.h"

#include "mnm/ir.h"
#include "mnm/value.h"
#include "mnm/registry.h"
#include "mnm/op.h"
#include "mnm/op_utils.h"
#include "mnm/memory_pool.h"

#include "../../../common/cuda_utils.h"

namespace mnm {
namespace op {
namespace cutlass {

using namespace ::cutlass;
using namespace ::cutlass::library;
using namespace mnm::ir;

class CutlassOpEnv : public mnm::op::OpEnv {
 public:
  explicit CutlassOpEnv(const CallValues& call);

  void Execute(const CallValues& call);

  /*! \brief Returns compute capability of the selected device. */
  int compute_capability() const;

  /*! \brief Request workspace. The workspace is allocated immediately. */
  void RequestWorkspace(void** dest, const Device& device, int64_t nbytes);

 protected:
  /*! \brief Host workspace */
  static int const kHostWorkspaceSize = (4 << 10);

  /*! \brief Provider of operations */
  Provider provider_{Provider::kCUTLASS};

  /*! \brief CUDA device properties */
  cudaDeviceProp device_prop_;

  /*! \brief CUDA device */
  Device device_;

  /*! \brief CUDA stream */
  cudaStream_t stream_{nullptr};

  /*! \brief Device workspace */
  void* workspace_{nullptr};

  /*! \brief Host workspace */
  char host_workspace_[kHostWorkspaceSize];

  /*! \brief Size of device workspace in bytes */
  size_t workspace_size_{0};

  /*! \brief Indicates whether scalars are host or device pointers */
  ScalarPointerMode scalar_pointer_mode_{ScalarPointerMode::kHost};

  /*! \brief Pointer to the operation to be executed */
  Operation const* operation_{nullptr};

  /*! \brief Device workspace memory */
  std::shared_ptr<memory_pool::Memory> workspace_mem_{nullptr};
};

NumericTypeID GetNumericTypeID(DType dtype);

std::vector<int> GetArgIndices(const op::CallValues& call, const Array<Var>& params);

template <typename T>
T GetPattern(const ir::Map<ir::DFPattern, ir::Array<ir::Expr>>& vmap, ir::DFPattern x) {
  if (vmap.count(x) == 0) {
    return T();
  }
  Array<Expr> value = vmap.at(x);
  CHECK_EQ(value.size(), 1U);
  return Downcast<T>(value[0]);
}

template <typename T>
T GetValue(const op::CallValues& call, const Var& var) {
  ir::Function func = ir::Downcast<value::ClosureValue>(call->callee)->func;
  ir::Array<value::Value> args = GetListArgs(call->args);
  size_t num = func->params.size();
  CHECK_EQ(args.size(), num);
  for (size_t i = 0; i < num; ++i) {
    if (func->params[i] == var) {
      return ir::Downcast<T>(args[i]);
    }
  }
  return T{};
}

}  // namespace cutlass
}  // namespace op
}  // namespace mnm

#define CUTLASS_CALL(func)                                                                     \
  do {                                                                                         \
    ::cutlass::Status status = (func);                                                         \
    CHECK(status == ::cutlass::Status::kSuccess) << ::cutlass::cutlassGetStatusString(status); \
  } while (false)
