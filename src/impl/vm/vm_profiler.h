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
  VirtualMachineProfiler(bool enable_cuda_graph, bool cache_interm_tensors)
      : VirtualMachine(enable_cuda_graph), cache_interm_tensors_(cache_interm_tensors) {
  }

  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final;

  ~VirtualMachineProfiler() {
  }

 protected:
  void ExecuteOpEnv(OpEnv* op_env, const std::vector<value::Value>& inputs,
                    value::Value output) final;

  std::shared_ptr<memory_pool::Memory> AllocCommon(const Device& dev, int64_t nbytes,
                                                   int64_t alignment, std::string memory_type);

  std::shared_ptr<memory_pool::Memory> AllocTensor(
      const Device& dev, int64_t nbytes, int64_t alignment = kDefaultMemoryAlignment) final;

  std::shared_ptr<memory_pool::Memory> AllocWorkspace(
      const Device& dev, int64_t nbytes, int64_t alignment = kDefaultMemoryAlignment) final;

 private:
  /*! \brief the duration of op call */
  std::unordered_map<OpEnv*, std::vector<double>> op_durations_;
  /*! \brief the input and output shape string of op call */
  std::unordered_map<OpEnv*, std::string> op_shapes_;
  /*! \brief the number of times of op call*/
  std::unordered_map<OpEnv*, int> op_invokes_;
  /*! \brief whether to cache intermediate tensors */
  bool cache_interm_tensors_;
  /*! \brief all op envs sorted in invoke order */
  Array<String> op_names_;
  /*! \brief the inputs for op_envs_ */
  Array<Array<Value>> op_inputs_;
  /*! \brief the outputs for op_envs_ */
  Array<Value> op_outputs_;
  /*! \brief whether to run memory profiling mode */
  bool profile_memory_ = false;
  /*! \brief map from memory type to total allocated size in MBs. */
  std::unordered_map<std::string, float> allocated_memory_mbs_;
};

}  // namespace vm
}  // namespace executor
}  // namespace mnm
