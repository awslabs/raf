/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/impl/vm/vm_debugguer.h
 * \brief The RAF virtual machine debugger.
 */
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "raf/vm/vm.h"

namespace raf {
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
}  // namespace raf
