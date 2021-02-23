/*!
 * \file src/impl/vm/vm_profiler.cc
 * \brief The Relay debug virtual machine.
 */

#include "vm_profiler.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "mnm/device_api.h"

namespace mnm {
namespace executor {
namespace vm {

PackedFunc VirtualMachineProfiler::GetFunction(const std::string& name,
                                               const ObjectPtr<Object>& sptr_to_self) {
  if (name == "get_stat") {
    return PackedFunc([sptr_to_self, this](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      ICHECK_EQ(args.size(), 1U);
      std::vector<std::pair<OpEnv*, double>> op_acc_time;
      for (auto kv : op_durations_) {
        auto val =
            std::make_pair(kv.first, std::accumulate(kv.second.begin(), kv.second.end(), 0.0));
        op_acc_time.push_back(val);
      }
      bool sort_by_time = args[0];
      if (sort_by_time) {
        auto comp = [](const std::pair<OpEnv*, double>& lhs, const std::pair<OpEnv*, double>& rhs) {
          return lhs.second > rhs.second;
        };
        std::sort(op_acc_time.begin(), op_acc_time.end(), comp);
      }
      double total_duration = 0.0;
      int64_t total_packed_funcs = 0;
      std::ostringstream os;
      os << std::setw(80) << std::left << "#OpName"
         << "\t" << std::setw(10) << std::left << "#InvokeCount"
         << "\t"
         << "#Duration(us): Sum/Mean/Min/Max" << std::endl;

      for (auto kv : op_acc_time) {
        auto vals = op_durations_[kv.first];
        auto sum = kv.second;
        auto mean = sum / static_cast<double>(vals.size());
        auto min_value = *std::min_element(vals.begin(), vals.end());
        auto max_value = *std::max_element(vals.begin(), vals.end());

        os << std::setw(80) << std::left << kv.first->env_name << "\t" << std::setw(10) << std::left
           << op_invokes_[kv.first] << "\t" << sum << "/" << mean << "/" << min_value << "/"
           << max_value << std::endl;

        total_duration += sum;
        total_packed_funcs += op_invokes_[kv.first];
      }
      os << "\nTotal Duration: " << total_duration << " us.\t"
         << "Total Packed Functions: " << total_packed_funcs << std::endl;
      *rv = os.str();
    });
  } else if (name == "reset") {
    return PackedFunc([sptr_to_self, this](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      op_durations_.clear();
      op_invokes_.clear();
    });
  } else {
    return VirtualMachine::GetFunction(name, sptr_to_self);
  }
}

void VirtualMachineProfiler::ExecuteOpEnv(OpEnv* op_env, const std::vector<value::Value>& inputs,
                                          value::Value output) {
  using namespace device_api;
  for (const auto& device : devices_) {
    const std::shared_ptr<DeviceAPI> device_api = DeviceAPI::Get(device.device_type);
    device_api->WaitDevice(device);
  }
  auto op_begin = std::chrono::high_resolution_clock::now();
  VirtualMachine::ExecuteOpEnv(op_env, inputs, output);
  for (const auto& device : devices_) {
    const std::shared_ptr<DeviceAPI> device_api = DeviceAPI::Get(device.device_type);
    device_api->WaitDevice(device);
  }
  auto op_end = std::chrono::high_resolution_clock::now();
  double op_duration =
      std::chrono::duration_cast<std::chrono::duration<double>>(op_end - op_begin).count();
  if (op_durations_.find(op_env) == op_durations_.end()) {
    CHECK(op_invokes_.find(op_env) == op_invokes_.end());
    op_durations_[op_env] = {};
    op_invokes_[op_env] = 0;
  }
  op_durations_[op_env].push_back(op_duration * 1e6);
  op_invokes_[op_env]++;
}

tvm::runtime::Module CreateVirtualMachineDebug(const Executable* exec, bool enable_cuda_graph) {
  auto vm = make_object<VirtualMachineProfiler>(enable_cuda_graph);
  vm->LoadExecutable(exec);
  return tvm::runtime::Module(vm);
}

MNM_REGISTER_GLOBAL("mnm.vm.VirtualMachineProfiler")
    .set_body([](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      tvm::runtime::Module mod = args[0];
      bool enable_cuda_graph = args[1];
      const auto* exec = dynamic_cast<Executable*>(mod.operator->());
      CHECK(exec) << "The virtual machine executable has not been defined yet.";
      *rv = CreateVirtualMachineDebug(exec, enable_cuda_graph);
    });

}  // namespace vm
}  // namespace executor
}  // namespace mnm
