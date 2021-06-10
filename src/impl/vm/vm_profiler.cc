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

#include "tvm/relay/transform.h"
#include "mnm/device_api.h"
#include "mnm/memory_pool.h"

namespace mnm {
namespace executor {
namespace vm {

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
  using PassContext = tvm::relay::transform::PassContext;
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
  } else if (name == "get_memory_trace") {
    return PackedFunc([sptr_to_self, this](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
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
        std::string name = memory_trace_[i].first->env_name;
        os << std::setw(6) << std::left << i << "\t" << std::setw(80) << std::left << name;
        for (auto trace : memory_trace_[i].second) {
          os << "\t" << std::setw(15) << std::left << trace.first;
          os << "\t" << std::setw(15) << std::left << trace.second;
        }
        os << std::endl;
      }
      *rv = os.str();
    });
  } else if (name == "run") {
    return PackedFunc([sptr_to_self, this](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      memory_trace_.clear();
      num_gcs_.clear();
      num_gcs_.resize(devices_.size());
      std::fill(num_gcs_.begin(), num_gcs_.end(), 0);
      peak_memory_mbs_.clear();
      peak_memory_mbs_.resize(devices_.size());
      std::fill(peak_memory_mbs_.begin(), peak_memory_mbs_.end(), std::make_pair(0, 0));
      profile_memory_ = false;
      if (args.size() == 2) {
        profile_memory_ = args[1];
      } else {
        ICHECK_EQ(args.size(), 1U);
      }

      VMContext ctx = args[0];
      if (profile_memory_) {
        for (auto device : devices_) {
          memory_pool::Memory::RemovePool(device);
          memory_pool::Memory::InitPool(device, memory_pool_name_);
        }

        PassContext::Current()->config.Set("mnm.tvmjit.allow_jit_failure", tvm::Bool(true));
        Run(ctx);
        PassContext::Current()->config.Set("mnm.tvmjit.allow_jit_failure", tvm::Bool(false));
        Map<String, ObjectRef> ret;
        for (size_t i = 0; i < devices_.size(); ++i) {
          const auto& device = devices_[i];
          std::string device_str = std::string(device.c_str());
          ret.Set(device_str,
                  Array<FloatImm>({FloatImm(DataType::Float(32), peak_memory_mbs_[i].first),
                                   FloatImm(DataType::Float(32), peak_memory_mbs_[i].second)}));
          ret.Set("GC_" + device_str, IntImm(DataType::Int(32), num_gcs_[i]));
          memory_pool::Memory::RemovePool(device);
        }
        *rv = ret;
      } else {
        *rv = Run(ctx);
      }
      profile_memory_ = false;
    });
  } else if (name == "reset") {
    return PackedFunc([sptr_to_self, this](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      op_durations_.clear();
      op_shapes_.clear();
      op_invokes_.clear();
      op_outputs_.clear();
      op_inputs_.clear();
      op_names_.clear();
      memory_trace_.clear();
      peak_memory_mbs_.clear();
      for (auto op_env_cache : op_env_cache_) {
        op_env_cache->Clear();
      }
    });
  } else {
    return VirtualMachine::GetFunction(name, sptr_to_self);
  }
}

void VirtualMachineProfiler::ExecuteOpEnv(OpEnv* op_env, const std::vector<value::Value>& inputs,
                                          value::Value output) {
  using namespace device_api;

  std::vector<std::pair<float, float>> curr_mems;
  for (size_t i = 0; i < devices_.size(); ++i) {
    const auto& device = devices_[i];
    auto curr_mem = memory_pool::Memory::GetPoolSize(device, memory_pool_name_);
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
  memory_trace_.push_back(std::make_pair(op_env, curr_mems));

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
