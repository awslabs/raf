/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file executor.h
 * \brief Executor API
 */
#pragma once
#include "./memory_pool.h"
#include "./op.h"
#include "./stream_pool.h"
#include "./value.h"
#include "ir.h"

namespace raf {
namespace requests {
class Requests;
}  // namespace requests
}  // namespace raf

namespace raf {
namespace executor {

class Executor {
 public:
  virtual ~Executor() = default;
  virtual void OnBind(const op::OpEnv* op_env) = 0;
  virtual void OnDestruct(const op::OpEnv* op_env) = 0;
  virtual void RequestWorkspace(requests::Requests* request, int index) = 0;
  virtual void RequestStream(requests::Requests* request, int index) = 0;
  virtual void RequestDistributed(requests::Requests* request, int index) = 0;
};

namespace interpreter {
value::Value Interpret(ir::Expr expr, ir::Optional<ir::IRModule> mod = {});
value::Value InvokePrimitive(const op::CallValues& call);
value::Value InvokeClosure(const op::CallValues& call);
}  // namespace interpreter
}  // namespace executor
}  // namespace raf
