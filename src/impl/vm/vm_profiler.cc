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

Value CopyTo(Value src, const Device& dev);

std::string GetShapeStr(const Value& value) {
  std::stringstream ss;
  ss << "(";
  if (const auto* tvo = value.as<TensorValueObj>()) {
    const auto& tensor = tvo->tensor;
    auto ndim = tensor->ndim;
    for (size_t i = 0; i < ndim; ++i) {
      ss << tensor->shape[i];
      if (ndim == 1 || i < ndim - 1) {
        ss << ",";
      }
    }
  } else if (const auto* tuple = value.as<TupleValueObj>()) {
    int size = static_cast<int>(tuple->fields.size());
    for (size_t i = 0; i < size; ++i) {
      if (i > 0) {
        ss << ",";
      }
      ss << GetShapeStr(tuple->fields[i]);
    }
  } else {
    ss << value->GetTypeKey();
  }
  ss << ")";
  return ss.str();
}

PackedFunc VirtualMachineProfiler::GetFunction(const std::string& name,
                                               const ObjectPtr<Object>& sptr_to_self) {
  using namespace device_api;
  if (name == "get_interm_tensors") {
    return PackedFunc([sptr_to_self, this](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      CHECK(cache_interm_tensors_)
          << "No intermediate tensor is cached. Please use "
          << "cache_interm_tensors=True to create the VirtualMachineProfiler";
      ICHECK_EQ(args.size(), 0U);
      CHECK_EQ(op_inputs_.size(), op_names_.size());
      CHECK_EQ(op_outputs_.size(), op_names_.size());
      Map<String, ObjectRef> res{
          {"names", op_names_}, {"inputs", op_inputs_}, {"outputs", op_outputs_}};
      *rv = res;
    });
  } else if (name == "get_stat") {
    return PackedFunc([sptr_to_self, this](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      ICHECK_EQ(args.size(), 2U);
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
      bool show_shape = args[1];
      auto title = (show_shape) ? "#OpNameNShape" : "#OpName";
      os << std::setw(80) << std::left << title << "\t" << std::setw(10) << std::left
         << "#InvokeCount"
         << "\t"
         << "#Duration(us): Sum/Mean/Min/Max" << std::endl;

      for (auto kv : op_acc_time) {
        std::string name = kv.first->env_name;
        if (show_shape) {
          name += " " + op_shapes_[kv.first];
        }
        auto vals = op_durations_[kv.first];
        auto sum = kv.second;
        auto mean = sum / static_cast<double>(vals.size());
        auto min_value = *std::min_element(vals.begin(), vals.end());
        auto max_value = *std::max_element(vals.begin(), vals.end());

        os << std::setw(80) << std::left << name << "\t" << std::setw(10) << std::left
           << op_invokes_[kv.first] << "\t" << sum << "/" << mean << "/" << min_value << "/"
           << max_value << std::endl;

        total_duration += sum;
        total_packed_funcs += op_invokes_[kv.first];
      }
      os << "\nTotal Duration: " << total_duration << " us.\t"
         << "Total Packed Functions: " << total_packed_funcs << std::endl;
      *rv = os.str();
    });
  } else if (name == "profile_memory") {
    return PackedFunc([sptr_to_self, this](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      ICHECK_EQ(args.size(), 1U);
      VMContext ctx = args[0];
      profile_memory_ = true;
      total_allocated_megabytes_ = 0;
      Run(ctx);
      profile_memory_ = false;
      for (auto op_env_cache : op_env_cache_) {
        op_env_cache->Clear();
      }
      *rv = total_allocated_megabytes_;
    });
  } else if (name == "reset") {
    return PackedFunc([sptr_to_self, this](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      op_durations_.clear();
      op_shapes_.clear();
      op_invokes_.clear();
      op_outputs_.clear();
      op_inputs_.clear();
      op_names_.clear();
      total_allocated_megabytes_ = 0.0;
    });
  } else {
    return VirtualMachine::GetFunction(name, sptr_to_self);
  }
}

std::tuple<std::shared_ptr<OpEnv>, std::vector<Value>, Value> VirtualMachineProfiler::PrepareOpEnv(
    const VMContext& ctx, const Instruction& instr) {
  if (profile_memory_) {
    // Skip the compilation in memory profiling mode.
    std::shared_ptr<OpEnv> op_env;
    std::vector<Value> inputs;
    Value output;
    return std::make_tuple(op_env, std::move(inputs), std::move(output));
  }
  return VirtualMachine::PrepareOpEnv(ctx, instr);
}

void VirtualMachineProfiler::ExecuteOpEnv(OpEnv* op_env, const std::vector<value::Value>& inputs,
                                          value::Value output) {
  using namespace device_api;
  if (profile_memory_) {
    // Skip execution in memory profiling mode.
    return;
  }

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

    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (i > 0) {
        ss << ",";
      }
      ss << GetShapeStr(inputs[i]);
    }
    ss << "]";
    ss << "," << GetShapeStr(output);
    op_shapes_[op_env] = ss.str();
  }
  op_durations_[op_env].push_back(op_duration * 1e6);
  op_invokes_[op_env]++;

  if (cache_interm_tensors_) {
    static Device cpu(DevType::kCPU(), 0);
    Array<Value> input;
    for (const auto& v : inputs) {
      input.push_back(CopyTo(v, cpu));
    }
    op_inputs_.push_back(input);
    op_outputs_.push_back(CopyTo(output, cpu));
    op_names_.push_back(op_env->env_name);
  }
}

std::shared_ptr<memory_pool::Memory> VirtualMachineProfiler::Alloc(const Device& dev,
                                                                   int64_t nbytes,
                                                                   int64_t alignment) {
  int64_t alloc_nbytes = memory_pool::Memory::GetAllocBytes(dev, nbytes);
  total_allocated_megabytes_ += alloc_nbytes / 1048576.0;
  if (profile_memory_) {
    // Allocate the minimum size to avoid out of memory during memory profiling.
    return memory_pool::Memory::Alloc(dev, 1);
  }
  return VirtualMachine::Alloc(dev, nbytes, alignment);
}

tvm::runtime::Module CreateVirtualMachineProfiler(const Executable* exec, bool enable_cuda_graph,
                                                  bool cache_interm_tensors) {
  auto vm = make_object<VirtualMachineProfiler>(enable_cuda_graph, cache_interm_tensors);
  vm->LoadExecutable(exec);
  return tvm::runtime::Module(vm);
}

MNM_REGISTER_GLOBAL("mnm.vm.VirtualMachineProfiler")
    .set_body([](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      tvm::runtime::Module mod = args[0];
      bool enable_cuda_graph = args[1];
      bool cache_interm_tensors = args[2];
      const auto* exec = dynamic_cast<Executable*>(mod.operator->());
      CHECK(exec) << "The virtual machine executable has not been defined yet.";
      *rv = CreateVirtualMachineProfiler(exec, enable_cuda_graph, cache_interm_tensors);
    });

}  // namespace vm
}  // namespace executor
}  // namespace mnm
