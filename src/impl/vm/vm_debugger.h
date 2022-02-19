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
 * \file src/impl/vm/vm_debugguer.h
 * \brief The Meta virtual machine debugger.
 */
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "mnm/vm/vm.h"

namespace mnm {
namespace executor {
namespace vm {

class VMDebugger : public VirtualMachine {
 public:
  VMDebugger() : VirtualMachine(false, false) {
  }

  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final;

 protected:
  void HandleInvokeJit(VMContext& ctx, const Instruction& instr) final;

 private:
  /*! \brief the number of times of op call */
  std::unordered_map<OpEnv*, int> op_invokes_;
  /*! \brief the input and output shape string of op call */
  std::unordered_map<OpEnv*, std::string> op_shapes_;
  /*! \brief all op envs sorted in invoke order */
  Array<String> op_names_;
  /*! \brief the inputs for op_envs_ */
  Array<Array<Value>> op_inputs_;
  /*! \brief the outputs for op_envs_ */
  Array<Value> op_outputs_;
};

}  // namespace vm
}  // namespace executor
}  // namespace mnm
