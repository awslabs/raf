/*!
 * \file src/impl/vm/vm_memory_profiler.cc
 * \brief The implementation for Meta VM memory profiler.
 */

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "mnm/device_api.h"
#include "mnm/memory_pool.h"
#include "mnm/pass.h"
#include "./vm_memory_profiler.h"
#include "../../requests.h"

namespace mnm {
namespace executor {
namespace vm {

PackedFunc VMMemoryProfiler::GetFunction(const std::string& name,
                                         const ObjectPtr<Object>& sptr_to_self) {
  if (name == "trace") {
    return PackedFunc([sptr_to_self, this](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      VMContext ctx = args[0];
      *rv = Trace(ctx);
    });
  } else if (name == "get_result") {
    return PackedFunc(
        [sptr_to_self, this](tvm::TVMArgs args, tvm::TVMRetValue* rv) { *rv = GetResult(); });
  } else {
    return VirtualMachine::GetFunction(name, sptr_to_self);
  }
}

Map<String, ObjectRef> VMMemoryProfiler::Trace(VMContext ctx) {
  // Reset the states and memory pool
  memory_trace_.clear();
  num_gcs_.clear();
  num_gcs_.resize(devices_.size());
  std::fill(num_gcs_.begin(), num_gcs_.end(), 0);
  peak_memory_mbs_.clear();
  peak_memory_mbs_.resize(devices_.size());
  std::fill(peak_memory_mbs_.begin(), peak_memory_mbs_.end(), std::make_pair(0, 0));
  for (auto op_env_cache : op_env_cache_) {
    op_env_cache->Clear();
  }
  for (auto device : devices_) {
    memory_pool::Memory::ResetPool(device);
  }

  // Run the vm
  pass::PassContext::Current()->config.Set("mnm.tvm.allow_jit_failure", tvm::Bool(true));
  VirtualMachine::Run(ctx);
  pass::PassContext::Current()->config.Set("mnm.tvm.allow_jit_failure", tvm::Bool(false));

  // Prepare the return
  Map<String, ObjectRef> ret;
  for (size_t i = 0; i < devices_.size(); ++i) {
    const auto& device = devices_[i];
    std::string device_str = std::string(device.c_str());
    ret.Set(device_str,
            Array<FloatImm>({FloatImm(DataType::Float(32), peak_memory_mbs_[i].first),
                             FloatImm(DataType::Float(32), peak_memory_mbs_[i].second)}));
    ret.Set("GC_" + device_str, IntImm(DataType::Int(32), num_gcs_[i]));
  }
  return ret;
}

std::string VMMemoryProfiler::GetResult() const {
  std::ostringstream os;

  // Display the memory trace of used memory instead of the total allocated memory.
  os << "Numbers are the in MBs." << std::endl;
  os << std::setw(6) << std::left << "#Trace\t" << std::setw(80) << std::left << "#OpName";

  for (const auto& device : devices_) {
    auto device_str = "#" + std::string(device.c_str());
    os << "\t" << std::setw(15) << std::left << device_str + "_used";
    os << "\t" << std::setw(15) << std::left << device_str + "_alloc";
  }
  os << std::endl;

  for (size_t i = 0; i < memory_trace_.size(); ++i) {
    std::string name = memory_trace_[i].first->name();
    os << std::setw(6) << std::left << i << "\t" << std::setw(80) << std::left << name;
    for (auto trace : memory_trace_[i].second) {
      os << "\t" << std::setw(15) << std::left << trace.first;
      os << "\t" << std::setw(15) << std::left << trace.second;
    }
    os << std::endl;
  }
  return os.str();
}

void VMMemoryProfiler::HandleInvokeJit(VMContext& ctx, const Instruction& instr) {
  OpEnvPtr op_env;
  std::vector<Value> inputs;
  Value output;
  std::string input_str;

  std::tie(op_env, inputs, output, input_str) = PrepareOpEnv(ctx, instr);
  // no need to run the op_env in the memory trace

  std::vector<std::pair<float, float>> curr_mems;
  for (size_t i = 0; i < devices_.size(); ++i) {
    const auto& device = devices_[i];
    auto curr_mem = memory_pool::Memory::GetPoolSize(device);
    curr_mems.push_back(curr_mem);

    float curr_used = 0, curr_total = 0;
    std::tie(curr_used, curr_total) = curr_mem;
    float peak_used = 0, peak_total = 0;
    std::tie(peak_used, peak_total) = peak_memory_mbs_[i];
    if (!memory_trace_.empty() && curr_total < memory_trace_.back().second[i].second) {
      // GC was triggered if the current allocated memory is smaller than executing the previous op.
      num_gcs_[i]++;
    }
    peak_memory_mbs_[i] = std::make_pair((peak_used < curr_used) ? curr_used : peak_used,
                                         (peak_total < curr_total) ? curr_total : peak_total);
  }
  memory_trace_.emplace_back(op_env, curr_mems);

  // Release workspace memory.
  // TODO(yaoyaoding): It seems that we can not release the workspace once we launched the
  //   kernel. Because the kernel may be in the executing status at this point due to
  //   asynchronous execution. This would cause problem for multi-stream execution.
  std::shared_ptr<requests::Requests> requests = op_env->GetRequests();
  for (size_t i = 0; i < requests->workspace.size(); ++i) {
    requests::Requests::WorkspaceRequest& entry = requests->workspace[i];
    if (entry.nbytes > 0 && entry.memory != nullptr) {
      *entry.dest = nullptr;
      entry.memory.reset();
    }
  }

  ctx->pc++;
}

tvm::runtime::Module CreateVMMemoryProfiler(const Executable* exec) {
  auto vm = make_object<VMMemoryProfiler>();
  vm->LoadExecutable(exec);
  return tvm::runtime::Module(vm);
}

MNM_REGISTER_GLOBAL("mnm.vm.VMMemoryProfiler")
    .set_body([](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      tvm::runtime::Module mod = args[0];
      const auto* exec = dynamic_cast<Executable*>(mod.operator->());
      CHECK(exec) << "The virtual machine executable has not been defined yet.";
      *rv = CreateVMMemoryProfiler(exec);
    });

}  // namespace vm
}  // namespace executor
}  // namespace mnm
