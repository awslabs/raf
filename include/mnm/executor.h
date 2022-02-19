/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
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
}  // namespace mnm
