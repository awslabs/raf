/*!
 * Copyright (c) 2020 by Contributors
 * \file src/impl/vm/vm.cc
 * \brief The Meta virtual machine.
 */

#include <dmlc/memory_io.h>
#include <tvm/runtime/container.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/device_api.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "mnm/memory_pool.h"
#include "mnm/ir.h"
#include "mnm/op.h"
#include "mnm/value.h"
#include "mnm/vm/bytecode.h"
#include "mnm/vm/vm.h"
#include "mnm/device_api.h"
#include "mnm/profiler.h"
#include "../../requests.h"

#include "mnm/device_api.h"
#include "mnm/registry.h"

#ifdef MNM_USE_CUDA
#include "../../common/cuda_utils.h"
#include "../../op/dispatch/cudnn/cudnn_utils.h"
#include "../../op/dispatch/cublas/cublas_utils.h"
#endif

namespace mnm {
namespace executor {
namespace vm {

using namespace mnm::ir;
using namespace mnm::value;
using namespace mnm::op;
using namespace mnm::registry;
using namespace mnm::requests;
using namespace mnm::device_api;

inline Value CopyTo(Value src, const Device& dev) {
  if (!src.defined()) {
    return src;
  }
  if (src.as<TensorValueObj>()) {
    auto tensor = Downcast<TensorValue>(src)->tensor;
    if (tensor->ctx.device_type != dev.device_type) {
      return TensorValue::make(tensor::Tensor(tensor.CopyTo(dev)));
    }
    return src;
  }
  if (src.as<TupleValueObj>()) {
    std::vector<Value> ret;
    TupleValue tup = Downcast<TupleValue>(src);
    for (size_t i = 0; i < tup->fields.size(); ++i) {
      ret.push_back(CopyTo(tup->fields[i], dev));
    }
    return TupleValue::make(ret);
  }
  return src;
}

MNM_REGISTER_OBJECT_REFLECT(VMContextObj);

VMContext VMContext::make(const Executable* exec) {
  auto ptr = make_object<VMContextObj>();
  ptr->exec = exec;
  return VMContext(ptr);
}

inline Value VMContext::ReadRegister(Index reg) const {
  auto self = this->operator->();
  return self->frames.back().register_file[reg];
}

inline void VMContext::WriteRegister(Index reg, const Value& val) {
  auto self = this->operator->();
  self->frames.back().register_file[reg] = val;
}

inline int64_t VMContext::LoadScalarInt(Index r) const {
  int32_t result;
  const auto& obj = ReadRegister(r);
  auto int_value = Downcast<IntValue>(obj);
  return int_value->value;
}

inline bool VMContext::IsConst(Index reg) const {
  auto self = this->operator->();
  return self->frames.back().is_const[reg];
}

inline void VMContext::PushFrame(Index func_index, const std::vector<Value>& args,
                                 RegName ret_reg) {
  auto self = this->operator->();
  const auto& func = self->exec->functions[func_index];
  CHECK_EQ(func.params.size(), args.size())
      << "Number of arguments mismatches: " << func.params.size() << " vs " << args.size();
  auto ret_pc = self->pc + 1;
  auto frame = VMFrame(self->func_index, ret_pc, ret_reg, args.size(), func.register_file_size);
  self->frames.push_back(frame);
  for (size_t i = 0; i < args.size(); ++i) {
    WriteRegister(i, args[i]);
  }
  self->func_index = func_index;
  self->code = func.instructions.data();
  self->pc = 0;
}

inline Index VMContext::PopFrame() {
  auto self = this->operator->();
  CHECK_GT(self->frames.size(), 0);
  const VMFrame& fr = self->frames.back();
  self->func_index = fr.caller_func_index;
  self->pc = fr.caller_return_pc;
  self->code = self->exec->functions[self->func_index].instructions.data();
  self->frames.pop_back();
  return fr.caller_return_register;
}

std::shared_ptr<OpEnvCache> VMFuncOpEnvCache::Get(Index pc) {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = cache_map_.find(pc);
  if (it != cache_map_.end()) {
    return it->second;
  }
  auto cache = std::make_shared<OpEnvCache>();
  cache_map_.emplace(pc, cache);
  return cache;
}

void VMFuncOpEnvCache::Clear() {
  std::lock_guard<std::mutex> lock(mu_);
  cache_map_.clear();
}

#ifdef MNM_USE_CUDA
void MNMSetStream(Device dev, cudaStream_t stream) {
  tvm::runtime::DeviceAPI::Get(dev)->SetStream(dev, stream);
  mnm::op::cudnn::SetStream(stream);
  mnm::op::cublas::SetStream(stream);
}

class VirtualMachine::CudaGraphImpl {
 public:
  CudaGraphImpl(Device dev) : device_(dev) {
    DLOG(INFO) << "Use Cuda Graph";
  }

  ~CudaGraphImpl() {
    CUDA_CALL(cudaGraphDestroy(graph_));
    CUDA_CALL(cudaGraphExecDestroy(exec_));
    CUDA_CALL(cudaStreamDestroy(stream_for_graph_));
  }

  void GetKernelInfo() {
    cudaGraphNode_t* nodes = NULL;
    size_t numNodes = 0;
    CUDA_CALL(cudaGraphGetNodes(graph_, nodes, &numNodes));
    cudaKernelNodeParams* pNodeParams;
    DLOG(INFO) << "Num of nodes in captured graph: " << (numNodes);
    CHECK_GT(numNodes, 0) << "Generated CUDA Graph is empty";
  }

  void BeginCapture() {
    stream_for_graph_ = static_cast<cudaStream_t>(
        mnm::device_api::DeviceAPI::Get(device_.device_type)->CreateStream(device_));

    MNMSetStream(device_, stream_for_graph_);
    CUDA_CALL(cudaStreamBeginCapture(stream_for_graph_, cudaStreamCaptureModeRelaxed));
  }

  void EndCapture() {
    CUDA_CALL(cudaStreamEndCapture(stream_for_graph_, &graph_));
    CUDA_CALL(cudaGraphInstantiate(&exec_, graph_, NULL, NULL, 0));
    GetKernelInfo();
    is_captured_ = true;
  }

  void Invoke() {
    CUDA_CALL(cudaGraphLaunch(exec_, NULL));
    CUDA_CALL(cudaStreamSynchronize(stream_for_graph_));
  }

 private:
  bool is_captured_ = false;
  cudaStream_t stream_for_graph_;
  cudaGraph_t graph_;
  cudaGraphExec_t exec_;
  Device device_;
};
#endif

PackedFunc VirtualMachine::GetFunction(const std::string& name,
                                       const ObjectPtr<Object>& sptr_to_self) {
  if (name == "run") {
    return PackedFunc([sptr_to_self, this](registry::TVMArgs args, registry::TVMRetValue* rv) {
      VMContext ctx = args[0];
      *rv = Run(ctx);
    });
  } else if (name == "set_devices") {
    return PackedFunc([sptr_to_self, this](registry::TVMArgs args, registry::TVMRetValue* rv) {
      std::vector<Device> devices;
      for (int i = 0; i < args.size(); ++i) {
        DLContext dev = args[i];
        devices.push_back(dev);
      }
      this->SetDevices(devices);
    });
  } else if (name == "prepare_context") {
    return PackedFunc([sptr_to_self, this](registry::TVMArgs args, registry::TVMRetValue* rv) {
      CHECK(exec_) << "The executable is not loaded yet.";
      std::string func_name = args[0];
      std::vector<Value> inputs(args.size() - 1);
      for (size_t i = 1; i < args.size(); ++i) {
        inputs[i - 1] = args[i];
      }
      *rv = PrepareVMContext(func_name, inputs);
    });
  } else {
    LOG(FATAL) << "Unknown packed function: " << name;
    return PackedFunc([sptr_to_self, name](registry::TVMArgs args, registry::TVMRetValue* rv) {});
  }
}

void VirtualMachine::LoadExecutable(const Executable* exec) {
  CHECK(exec) << "The executable is not created yet.";
  exec_ = exec;
  for (int i = 0; i < exec_->functions.size(); ++i) {
    op_env_cache_.push_back(std::make_shared<VMFuncOpEnvCache>());
  }

  tvm::runtime::Module lib = exec_->lib;
  // Get the list of packed functions.
  CHECK(exec->primitive_map.empty() || lib.operator->())
      << "runtime module should have been built for primitive functions"
      << "\n";
  for (const auto& it : exec_->primitive_map) {
    const auto& packed_name = it.first;
    auto packed_index = static_cast<size_t>(it.second);
    if (packed_funcs_.size() <= packed_index) {
      packed_funcs_.resize(packed_index + 1);
    }
    tvm::runtime::PackedFunc pf = lib.GetFunction(packed_name, true);
    CHECK(pf != nullptr) << "Cannot find function in module: " << packed_name;
    packed_funcs_[packed_index] = pf;
  }
}

VMContext VirtualMachine::PrepareVMContext(const std::string& func_name,
                                           const std::vector<Value>& inputs) {
  auto gvit = exec_->global_map.find(func_name);
  CHECK(gvit != exec_->global_map.end()) << "Cannot find function " << func_name;
  auto func_index = gvit->second;
  const auto& vm_func = exec_->functions[func_index];
  CHECK_EQ(inputs.size(), vm_func.params.size())
      << "The number of inputs doesn't match the number of parameters for function " << func_name;

  auto fcreate_ctx = [&]() {
    auto ctx = VMContext::make(exec_);
    ctx->entry_func_index = func_index;
    ctx->inputs.resize(inputs.size());
    // TODO(@zhiics, @icemelon9): For heterogeneous execution, get input device information
    Device dev = devices_[0];
    for (size_t i = 0; i < inputs.size(); ++i) {
      ctx->inputs[i] = CopyTo(inputs[i], dev);
    }
    return ctx;
  };
#ifdef MNM_USE_CUDA
  if (enable_cuda_graph_) {
    std::lock_guard<std::mutex> lock(cuda_graph_mutex_);
    // Check if there is another context using the CUDA graph
    CHECK(!cuda_graph_occupied_) << "VM in CUDA graph mode doesn't support concurrent execution";
    if (!cuda_graph_ctx_.defined() || cuda_graph_ctx_->entry_func_index != func_index) {
      // Initialize the cuda graph context for the first time, or reset the cuda graph context
      // because this time invokes a different function
      cuda_graph_impl_ = nullptr;
      cuda_graph_ctx_ = fcreate_ctx();
    } else {
      for (int i = 0; i < inputs.size(); i++) {
        Value new_arg = inputs[i];
        Value graph_arg = cuda_graph_ctx_->inputs[i];
        if (new_arg.as<TensorValueObj>()) {
          CHECK(graph_arg.as<TensorValueObj>()) << "Value type mismatch, cannot copy";
          Downcast<TensorValue>(new_arg)->tensor.CopyTo(Downcast<TensorValue>(graph_arg)->tensor);
        } else {
          LOG(FATAL) << "Unsupported Value Type for reusing CUDA Graph";
        }
      }
      DLOG(INFO) << "Updated the inputs to the cached CUDA Graph.";
    }
    cuda_graph_occupied_ = true;
    return cuda_graph_ctx_;
  }
#endif
  auto ctx = fcreate_ctx();
  return ctx;
}

Value VirtualMachine::Run(VMContext ctx) {
  auto frun = [&]() {
    ctx.PushFrame(ctx->entry_func_index, ctx->inputs, -1);
    RunLoop(ctx);
  };
#ifdef MNM_USE_CUDA
  if (enable_cuda_graph_) {
    CHECK(ctx.get() == cuda_graph_ctx_.get()) << "Wrong VMContext provided for CUDA graph.";
    if (!cuda_graph_impl_) {
      cuda_graph_impl_ = new CudaGraphImpl(devices_[0]);
      DLOG(INFO) << "Begin capturing CUDA graph.";
      cuda_graph_impl_->BeginCapture();
      frun();
      cuda_graph_impl_->EndCapture();
      DLOG(INFO) << "CUDA graph captured.";
    }
    cuda_graph_impl_->Invoke();
    std::lock_guard<std::mutex> lock(cuda_graph_mutex_);
    cuda_graph_occupied_ = false;
    // TODO(@icemelon9, @zhiics): May need to copy the return register to the host device to
    // avoid data race
    return ctx->return_register;
  }
#endif
  frun();
  return ctx->return_register;
}

Device VirtualMachine::GetParamsDevice() const {
  CHECK(!devices_.empty()) << "Devices have not been initialized yet.";

  // Use the fallback device if no device index is available.
  int fallback_device_type = static_cast<int>(devices_[0].device_type);
  // TODO(@zhiics): For heterogeneous execution, get device information from byte

  const auto& cit =
      std::find_if(devices_.begin(), devices_.end(), [&fallback_device_type](const Device& d) {
        return fallback_device_type == static_cast<int>(d.device_type);
      });
  return (cit == devices_.end() ? devices_[0] : *cit);
}

void VirtualMachine::SetDevices(const std::vector<Device>& devices) {
  devices_ = devices;
  bool has_gpu = false;
  for (const Device& dev : devices) {
    if (dev.device_type == DevType::kCUDA()) {
      has_gpu = true;
      break;
    }
  }
  if (!has_gpu) {
    enable_cuda_graph_ = false;
  }
}

void VirtualMachine::RunLoop(VMContext ctx) {
  CHECK(this->exec_);
  CHECK_GT(ctx->frames.size(), 0) << "The call stack is empty";
  CHECK(ctx->code);
  ctx->pc = 0;
  while (true) {
  main_loop:
    auto const& instr = ctx->code[ctx->pc];
#if USE_RELAY_DEBUG
    InstructionPrint(std::cout, instr);
#endif  // USE_RELAY_DEBUG

    switch (instr.op) {
      case Opcode::Move: {
        Value from_obj = ctx.ReadRegister(instr.from);
        ctx.WriteRegister(instr.dst, from_obj);
        ctx->pc++;
        goto main_loop;
      }
      case Opcode::Fatal: {
        throw std::runtime_error("VM encountered fatal error");
      }
      case Opcode::LoadConst: {
        auto constant_obj = exec_->constants[instr.const_index];
        // We cache the allocated object in the constant pool. To measure, the
        // first iteration will set the pool up. The other iterations will
        // directly reuse the allocated objects.
        if (const_pool_.size() <= static_cast<size_t>(instr.const_index)) {
          const_pool_.resize(instr.const_index + 1);
        }

        if (!const_pool_[instr.const_index].defined()) {
          // TODO(@zhiics): device could be obtained from the device list.
          const_pool_[instr.const_index] = CopyTo(constant_obj, devices_[0]);
        }
        ctx.WriteRegister(instr.dst, const_pool_[instr.const_index]);
        ctx->frames.back().is_const[instr.dst] = true;
        ctx->pc++;
        goto main_loop;
      }
      case Opcode::LoadConsti: {
        ctx.WriteRegister(instr.dst, ScalarValue::make(instr.load_consti.val));
        ctx->pc++;
        goto main_loop;
      }
      case Opcode::InvokeFunc: {
        std::vector<Value> args;
        for (Index i = 0; i < instr.invoke_func.num_args; ++i) {
          args.push_back(ctx.ReadRegister(instr.invoke_func.args[i]));
        }
        ctx.PushFrame(instr.invoke_func.func_index, args, instr.dst);
        goto main_loop;
      }
      case Opcode::InvokePacked: {
        LOG(FATAL) << "Not supported.";
      }
      case Opcode::InvokeClosure: {
        auto closure = Downcast<VMClosureValue>(ctx.ReadRegister(instr.invoke_closure.closure));
        std::vector<Value> args;
        for (auto free_var : closure->free_vars) {
          args.push_back(free_var);
        }
        for (Index i = 0; i < instr.invoke_closure.num_args; ++i) {
          args.push_back(ctx.ReadRegister(instr.invoke_closure.args[i]));
        }
        ctx.PushFrame(closure->func_index, args, instr.dst);
        goto main_loop;
      }
      case Opcode::GetField: {
        auto object = ctx.ReadRegister(instr.get_field.object);
        const auto& tuple = Downcast<TupleValue>(object);
        auto field = tuple->fields[instr.get_field.field_index];
        ctx.WriteRegister(instr.dst, field);
        ctx->pc++;
        goto main_loop;
      }
      case Opcode::Goto: {
        ctx->pc += instr.pc_offset;
        goto main_loop;
      }
      case Opcode::If: {
        int32_t test_val = ctx.LoadScalarInt(instr.if_op.test);
        int32_t target_val = ctx.LoadScalarInt(instr.if_op.target);

        if (test_val == target_val) {
          CHECK_NE(instr.if_op.true_offset, 0);
          ctx->pc += instr.if_op.true_offset;
        } else {
          CHECK_NE(instr.if_op.false_offset, 0);
          ctx->pc += instr.if_op.false_offset;
        }

        goto main_loop;
      }
      case Opcode::AllocTensor: {
        auto shape = std::vector<int64_t>(instr.alloc_tensor.ndim);

        for (uint32_t i = 0; i < instr.alloc_tensor.ndim; ++i) {
          shape[i] = instr.alloc_tensor.shape[i];
        }

        auto storage_obj = ctx.ReadRegister(instr.alloc_tensor.storage);
        auto storage = Downcast<StorageValue>(storage_obj);
        auto tensor = TensorValue::Assemble(storage->buffer->device, instr.alloc_tensor.dtype,
                                            shape, {}, storage->buffer->data, storage->buffer);
        ctx.WriteRegister(instr.dst, tensor);
        ctx->pc++;
        goto main_loop;
      }
      case Opcode::AllocTensorReg: {
        LOG(FATAL) << "Not supported";
      }
      case Opcode::AllocTuple: {
        Array<Value> fields;
        for (Index i = 0; i < instr.alloc_tuple.num_fields; ++i) {
          fields.push_back(ctx.ReadRegister(instr.alloc_tuple.fields[i]));
        }
        ctx.WriteRegister(instr.dst, TupleValue::make(fields));
        ctx->pc++;
        goto main_loop;
      }
      case Opcode::AllocClosure: {
        std::vector<Value> free_vars;
        for (Index i = 0; i < instr.alloc_closure.num_free_vars; i++) {
          free_vars.push_back(ctx.ReadRegister(instr.alloc_closure.free_vars[i]));
        }
        auto clo = VMClosureValue::make(instr.alloc_closure.func_index, free_vars);
        ctx.WriteRegister(instr.dst, clo);
        ctx->pc++;
        goto main_loop;
      }
      case Opcode::AllocStorage: {
        auto size = ctx.LoadScalarInt(instr.alloc_storage.allocation_size);
        auto alignment = instr.alloc_storage.alignment;

        DLOG(INFO) << "AllocStorage: allocation_size=" << size << " alignment=" << alignment
                   << " dtype_hint="
                   << tvm::runtime::DLDataType2String(instr.alloc_storage.dtype_hint);

        auto dev = Device(instr.alloc_storage.device_type, instr.alloc_storage.device_id);
        auto buffer = Alloc(dev, size);
        auto storage = StorageValue::make(buffer);
        ctx.WriteRegister(instr.dst, storage);
        ctx->pc++;
        goto main_loop;
      }
      case Opcode::Ret: {
        // If we have hit the point from which we started
        // running, we should return to the caller breaking
        // the dispatch loop.
        auto ret_val = ctx.ReadRegister(instr.result);
        auto caller_return_register = ctx.PopFrame();

        if (caller_return_register < 0) {
          ctx->return_register = ret_val;
          return;
        } else {  // Otherwise we are just returning from a local call.
          ctx.WriteRegister(caller_return_register, ret_val);
          goto main_loop;
        }
      }
      case Opcode::InvokeJit: {
        std::shared_ptr<OpEnv> op_env;
        std::vector<Value> inputs;
        Value output;

        std::tie(op_env, inputs, output) = PrepareOpEnv(ctx, instr);
        ExecuteOpEnv(op_env.get(), inputs, output);
        ctx->pc++;
        goto main_loop;
      }
    }
  }
}

tvm::runtime::Module CreateVirtualMachine(const Executable* exec, bool enable_cuda_graph) {
  auto vm = make_object<VirtualMachine>(enable_cuda_graph);
  vm->LoadExecutable(exec);
  return tvm::runtime::Module(vm);
}

std::tuple<std::shared_ptr<OpEnv>, std::vector<Value>, Value> VirtualMachine::PrepareOpEnv(
    const VMContext& ctx, const Instruction& instr) {
  Index num_inputs = instr.invoke_jit.arity - instr.invoke_jit.output_size;
  HashKey key;
  Array<Value> args;
  Value output;

  // extract the input args and prepare the hash key to query op env
  for (Index i = 0; i < num_inputs; i++) {
    Index reg_idx = instr.invoke_jit.args[i];
    auto reg = ctx.ReadRegister(reg_idx);
    args.push_back(reg);
    if (!ctx.IsConst(reg_idx)) {
      if (auto t = reg.as<TensorValueObj>()) {
        key << GetRef<TensorValue>(t);
      } else if (auto tup = reg.as<TupleValueObj>()) {
        for (auto field : tup->fields) {
          auto t = field.as<TensorValueObj>();
          CHECK(t != nullptr);
          key << GetRef<TensorValue>(t);
        }
      } else {
        LOG(FATAL) << "Unsupported non-const register type: " << reg->GetTypeKey();
      }
    }
  }

  // extract the output
  if (instr.invoke_jit.output_size == 1) {
    output = ctx.ReadRegister(instr.invoke_jit.args[num_inputs]);
  } else {
    Array<Value> outs;
    for (Index i = num_inputs; i < instr.invoke_jit.arity; i++) {
      outs.push_back(ctx.ReadRegister(instr.invoke_jit.args[i]));
    }
    output = TupleValue::make(outs);
  }

  // check the OpEnv cache
  std::shared_ptr<OpEnv> op_env;
  auto op_env_cache = op_env_cache_[ctx->func_index]->Get(ctx->pc);
  if (auto p = op_env_cache->Get(key.byte_vector)) {
    // Cache hit. Reuse the OpEnv from the cache.
    op_env = *p;
  } else {
    // Create a new OpEnv.
    static auto fschema = Op::GetAttrMap<op::FMNMSchema>("FMNMSchema");
    auto call_values = CallValues::make();
    Value callee = ctx.ReadRegister(instr.invoke_jit.op_reg);
    const auto* op = callee.as<OpValueObj>();
    const auto* closure = callee.as<ClosureValueObj>();
    call_values->callee = callee;
    if (op) {
      call_values->args = fschema[op->op](args);
    } else {
      call_values->args = MakeListArgs(args);
    }
    call_values->device = devices_[0];
    call_values->out = output;
    op_env = Dispatch(call_values);
    CHECK(op_env != nullptr) << "ValueError: Cannot dispatch "
                             << " @ " << call_values->device.c_str()
                             << (op ? op->op->name : PrettyPrint(closure->func));
    // TODO(vinx13): request stream
    std::shared_ptr<Requests> requests = op_env->GetRequests();
    for (size_t i = 0; i < requests->workspace.size(); i++) {
      Requests::WorkspaceRequest& entry = requests->workspace[i];
      auto buf = memory_pool::Memory::Alloc(entry.device, entry.nbytes);
      *entry.dest = buf->data;
    }
    // add to cache
    op_env_cache->Set(key.byte_vector, op_env);
  }

  std::vector<Value> inputs;
  for (int i : op_env->arg_indices) {
    CHECK_GE(i, 0) << "Invalid input index: " << i;
    inputs.push_back(args[i]);
  }
  return std::make_tuple(op_env, std::move(inputs), std::move(output));
}

void VirtualMachine::ExecuteOpEnv(OpEnv* op_env, const std::vector<value::Value>& inputs,
                                  value::Value output) {
  op_env->Execute(inputs, output);
}

std::shared_ptr<memory_pool::Memory> VirtualMachine::Alloc(const Device& dev, int64_t nbytes,
                                                           int64_t alignment) {
  return memory_pool::Memory::Alloc(dev, nbytes);
}

MNM_REGISTER_GLOBAL("mnm.vm.VirtualMachine").set_body([](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
  tvm::runtime::Module mod = args[0];
  bool enable_cuda_graph = args[1];
  const auto* exec = dynamic_cast<Executable*>(mod.operator->());
  CHECK(exec) << "The virtual machine executable has not been defined yet.";
  *rv = CreateVirtualMachine(exec, enable_cuda_graph);
});

}  // namespace vm
}  // namespace executor
}  // namespace mnm
