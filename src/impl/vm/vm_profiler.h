/*!
 * \file src/impl/vm/vm_profiler.h
 * \brief The Relay debug virtual machine.
 */

#pragma once

#include "mnm/vm/vm.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace mnm {
namespace executor {
namespace vm {

class VirtualMachineProfiler : public VirtualMachine {
 public:
  VirtualMachineProfiler(bool enable_cuda_graph) : VirtualMachine(enable_cuda_graph) {
  }

  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final;

  ~VirtualMachineProfiler() {
  }

 protected:
  void ExecuteOpEnv(OpEnv* op_env, const std::vector<value::Value>& inputs,
                    value::Value output) final;

 private:
  std::unordered_map<OpEnv*, std::vector<double>> op_durations_;
  std::unordered_map<OpEnv*, int> op_invokes_;
};

}  // namespace vm
}  // namespace executor
}  // namespace mnm
