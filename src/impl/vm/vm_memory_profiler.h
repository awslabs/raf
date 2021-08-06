/*!
 * \file src/impl/vm/vm_memory_profiler.h
 * \brief The Meta VM memory profiler.
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

class VMMemoryProfiler : public VirtualMachine {
 public:
  VMMemoryProfiler() : VirtualMachine(false) {
  }
  /*! \brief Get a PackedFunc from module. */
  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final;
  /*!
   * \brief Trace the memory use.
   * \param ctx The VM context.
   * \return Per device memory use.
   */
  Map<String, ObjectRef> Trace(VMContext ctx);
  /*! \brief Get the memory trace result. */
  std::string GetResult() const;

 protected:
  void HandleInvokeJit(VMContext& ctx, const Instruction& instr) final;

 private:
  /*! \brief The number of performed garbage collections for each device memory pool. */
  std::vector<size_t> num_gcs_;
  /*! \brief The peak of used memory and total allocated memory of each device memory pool */
  std::vector<std::pair<float, float>> peak_memory_mbs_;
  /*! \brief the memory trace that maps the executed op to peak memory of each device */
  std::vector<std::pair<OpEnvPtr, std::vector<std::pair<float, float>>>> memory_trace_;
};

}  // namespace vm
}  // namespace executor
}  // namespace mnm
