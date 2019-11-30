/*!
 * Copyright (c) 2019 by Contributors
 * \file executor.h
 * \brief Executor API
 */
#pragma once
#include "./memory_pool.h"
#include "./op.h"
#include "./stream_pool.h"
#include "./value.h"

namespace mnm {
namespace requests {
class Requests;
}  // namespace requests
}  // namespace mnm

namespace mnm {
namespace executor {

class Executor {
 public:
  virtual ~Executor() = default;
  virtual void OnBind(const op::OpEnv* op_env) = 0;
  virtual void OnDestruct(const op::OpEnv* op_env) = 0;
  virtual void RequestMemory(requests::Requests* request, int index) = 0;
  virtual void RequestWorkspace(requests::Requests* request, int index) = 0;
  virtual void RequestStream(requests::Requests* request, int index) = 0;
  virtual void RequestDistributed(requests::Requests* request, int index) {
    LOG(FATAL) << "NotImplementedError: RequestDistributed";
    throw;
  }
};

namespace interpreter {
ir::ObjectRef Interpret(ir::Expr expr, ir::Module mod = ir::NullValue<ir::Module>());
}  // namespace interpreter
}  // namespace executor
}  // namespace mnm
