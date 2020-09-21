/*!
 * Copyright (c) 2020 by Contributors
 * \file src/impl/vm/vm.cc
 * \brief The Meta virtual machine.
 */

#include <dmlc/memory_io.h>
#include <tvm/support/logging.h>
#include <tvm/runtime/container.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/object.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "mnm/memory_pool.h"
#include "mnm/ir.h"
#include "mnm/op.h"
#include "mnm/value.h"
#include "mnm/vm/vm.h"
#include "../../requests.h"

namespace mnm {
namespace executor {
namespace vm {

using namespace mnm::ir;
using namespace mnm::value;
using namespace mnm::op;
using namespace mnm::registry;
using namespace mnm::requests;

inline StorageValue make_storage(size_t size, size_t alignment, DLDataType dtype_hint,
                                 Context ctx) {
  auto buffer = memory_pool::Memory::Alloc(ctx, size);
  return StorageValue::make(buffer);
}

inline Value CopyTo(Value src, const DLContext& ctx) {
  if (!src.defined()) {
    return src;
  }
  if (src.as<TensorValueObj>()) {
    auto tensor = Downcast<TensorValue>(src)->tensor;
    if (tensor->ctx.device_type != ctx.device_type) {
      return TensorValue::make(tensor::Tensor(tensor.CopyTo(ctx)));
    }
    return src;
  }
  if (src.as<TupleValueObj>()) {
    std::vector<Value> ret;
    TupleValue tup = Downcast<TupleValue>(src);
    for (size_t i = 0; i < tup->fields.size(); ++i) {
      ret.push_back(CopyTo(tup->fields[i], ctx));
    }
    return TupleValue::make(ret);
  }
  return src;
}

PackedFunc VirtualMachine::GetFunction(const std::string& name,
                                       const ObjectPtr<Object>& sptr_to_self) {
  if (name == "invoke") {
    return PackedFunc([sptr_to_self, this](registry::TVMArgs args, registry::TVMRetValue* rv) {
      CHECK(exec_) << "The executable is not created yet.";
      std::string func_name = args[0];
      auto git = exec_->global_map.find(func_name);
      CHECK(git != exec_->global_map.end())
          << "Cannot find function " << func_name << " in the executable";
      auto func = exec_->functions[git->second];
      if (func.params.empty()) {
        *rv = Invoke(func, {});
      } else {
        auto it = inputs_.find(func_name);
        CHECK(it != inputs_.end()) << "Input has not been set for function " << func_name;
        const std::vector<Value>& func_args = it->second;
        *rv = Invoke(func, func_args);
      }
    });
  } else if (name == "init") {
    return PackedFunc([sptr_to_self, this](registry::TVMArgs args, registry::TVMRetValue* rv) {
      std::vector<Context> contexts;
      for (int i = 0; i < args.size(); ++i) {
        DLContext ctx = args[i];
        contexts.push_back(ctx);
      }
      this->Init(contexts);
    });
  } else if (name == "set_input") {
    return PackedFunc([sptr_to_self, this](registry::TVMArgs args, registry::TVMRetValue* rv) {
      CHECK(exec_) << "The executable is not created yet.";
      std::string func_name = args[0];
      auto gvit = exec_->global_map.find(func_name);
      CHECK(gvit != exec_->global_map.end()) << "Cannot find function " << func_name;
      auto func_index = gvit->second;
      const auto& vm_func = exec_->functions[func_index];
      const auto& param_names = vm_func.params;
      // TODO(icemelon9): For heterogeneous execution, get input device information
      TVMContext ctx = ctxs_[0];
      CHECK_EQ(args.size() - 1, param_names.size())
          << "The number of provided parameters doesn't match the number of arguments";
      std::vector<Value> func_args(param_names.size());
      for (int i = 1; i < args.size(); ++i) {
        Value arg = args[i];
        func_args[i - 1] = CopyTo(arg, ctx);
      }
      inputs_.erase(func_name);
      inputs_.emplace(func_name, func_args);
    });
  } else {
    LOG(FATAL) << "Unknown packed function: " << name;
    return PackedFunc([sptr_to_self, name](registry::TVMArgs args, registry::TVMRetValue* rv) {});
  }
}

void VirtualMachine::LoadExecutable(const Executable* exec) {
  CHECK(exec) << "The executable is not created yet.";
  exec_ = exec;

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

Context VirtualMachine::GetParamsContext() const {
  CHECK(!ctxs_.empty()) << "Context has not been initialized yet.";

  // Use the fallback device if no device index is available.
  int fallback_device_type = static_cast<int>(ctxs_[0].device_type);
  // TODO(wweic): For heterogeneous execution, get device information from byte

  const auto& cit =
      std::find_if(ctxs_.begin(), ctxs_.end(), [&fallback_device_type](const TVMContext& c) {
        return fallback_device_type == static_cast<int>(c.device_type);
      });
  return (cit == ctxs_.end() ? ctxs_[0] : *cit);
}

void VirtualMachine::PushFrame(Index arg_count, Index ret_pc, const VMFunction& vm_func) {
  auto frame = VMFrame(ret_pc, func_index_, arg_count, code_, vm_func.register_file_size);
  frames_.push_back(frame);
}

Index VirtualMachine::PopFrame() {
  CHECK_GT(frames_.size(), 0);
  const VMFrame& fr = frames_.back();
  func_index_ = fr.func_index;
  code_ = fr.code;
  pc_ = fr.pc;
  auto call_stack_size = frames_.size();
  frames_.pop_back();
  return call_stack_size;
}

void VirtualMachine::InvokeGlobal(const VMFunction& func, const std::vector<Value>& args) {
  DLOG(INFO) << "Invoking global " << func.name << " " << args.size();

  PushFrame(func.params.size(), this->pc_ + 1, func);
  for (size_t i = 0; i < args.size(); ++i) {
    WriteRegister(i, args[i]);
  }
  DLOG(INFO) << "func.params= " << func.params.size();

  code_ = func.instructions.data();
  pc_ = 0;
}

Value VirtualMachine::Invoke(const VMFunction& func, const std::vector<Value>& args) {
  DLOG(INFO) << "Executing Function: " << std::endl << func;

  InvokeGlobal(func, args);
  RunLoop();
  // TODO(wweic) ctx could be obtained from the ctxs list.
  // auto alloc = MemoryManager::Global()->GetAllocator(ctxs_[0]);
  // DLOG(INFO) << "Memory used: " << alloc->UsedMemory() << " B";
  return return_register_;
}

Value VirtualMachine::Invoke(const std::string& name, const std::vector<Value>& args) {
  CHECK(exec_) << "The executable has not been created yet.";
  auto it = exec_->global_map.find(name);
  CHECK(it != exec_->global_map.end()) << "Cannot find function " << name << " in the executable";
  auto func_index_ = it->second;
  DLOG(INFO) << "Invoke Global " << name << " at index " << func_index_;
  return Invoke(exec_->functions[func_index_], args);
}

void VirtualMachine::Init(const std::vector<Context>& ctxs) {
  ctxs_ = ctxs;
}

inline void VirtualMachine::WriteRegister(Index r, const Value& val) {
  frames_.back().register_file[r] = val;
}

inline Value VirtualMachine::ReadRegister(Index r) const {
  return frames_.back().register_file[r];
}

inline int64_t VirtualMachine::LoadScalarInt(Index r) const {
  int32_t result;
  const auto& obj = ReadRegister(r);
  auto int_value = Downcast<IntValue>(obj);
  return int_value->data;
}

void VirtualMachine::RunLoop() {
  CHECK(this->exec_);
  CHECK(this->code_);
  pc_ = 0;
  Index frame_start = frames_.size();
  while (true) {
  main_loop:
    auto const& instr = code_[this->pc_];
#if USE_RELAY_DEBUG
    InstructionPrint(std::cout, instr);
#endif  // USE_RELAY_DEBUG

    switch (instr.op) {
      case Opcode::Move: {
        Value from_obj = ReadRegister(instr.from);
        WriteRegister(instr.dst, from_obj);
        pc_++;
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
          // TODO(wweic) ctx could be obtained from the ctxs list.
          const_pool_[instr.const_index] = CopyTo(constant_obj, ctxs_[0]);
        }
        WriteRegister(instr.dst, const_pool_[instr.const_index]);
        pc_++;
        goto main_loop;
      }
      case Opcode::LoadConsti: {
        WriteRegister(instr.dst, IntValue::make(instr.load_consti.val));
        pc_++;
        goto main_loop;
      }
      case Opcode::InvokeFunc: {
        std::vector<Value> args;
        for (Index i = 0; i < instr.invoke_func.num_args; ++i) {
          args.push_back(ReadRegister(instr.invoke_func.args[i]));
        }
        InvokeGlobal(exec_->functions[instr.invoke_func.func_index], args);
        frames_.back().caller_return_register = instr.dst;
        goto main_loop;
      }
      case Opcode::InvokePacked: {
        LOG(FATAL) << "Not supported.";
      }
      case Opcode::InvokeClosure: {
        auto closure = Downcast<VMClosureValue>(ReadRegister(instr.invoke_closure.closure));
        std::vector<Value> args;
        for (auto free_var : closure->free_vars) {
          args.push_back(free_var);
        }
        for (Index i = 0; i < instr.invoke_closure.num_args; ++i) {
          args.push_back(ReadRegister(instr.invoke_closure.args[i]));
        }
        InvokeGlobal(exec_->functions[closure->func_index], args);
        frames_.back().caller_return_register = instr.dst;
        goto main_loop;
      }
      case Opcode::GetField: {
        auto object = ReadRegister(instr.get_field.object);
        const auto& tuple = Downcast<TupleValue>(object);
        auto field = tuple->fields[instr.get_field.field_index];
        WriteRegister(instr.dst, field);
        pc_++;
        goto main_loop;
      }
      case Opcode::Goto: {
        pc_ += instr.pc_offset;
        goto main_loop;
      }
      case Opcode::If: {
        int32_t test_val = LoadScalarInt(instr.if_op.test);
        int32_t target_val = LoadScalarInt(instr.if_op.target);

        if (test_val == target_val) {
          CHECK_NE(instr.if_op.true_offset, 0);
          pc_ += instr.if_op.true_offset;
        } else {
          CHECK_NE(instr.if_op.false_offset, 0);
          pc_ += instr.if_op.false_offset;
        }

        goto main_loop;
      }
      case Opcode::AllocTensor: {
        auto shape = std::vector<int64_t>(instr.alloc_tensor.ndim);

        for (uint32_t i = 0; i < instr.alloc_tensor.ndim; ++i) {
          shape[i] = instr.alloc_tensor.shape[i];
        }

        auto storage_obj = ReadRegister(instr.alloc_tensor.storage);
        auto storage = Downcast<StorageValue>(storage_obj);
        auto tensor = TensorValue::Assemble(ctxs_[0], instr.alloc_tensor.dtype, shape, {},
                                            storage->buffer->data);
        WriteRegister(instr.dst, tensor);
        pc_++;
        goto main_loop;
      }
      case Opcode::AllocTensorReg: {
        LOG(FATAL) << "Not supported";
      }
      case Opcode::AllocTuple: {
        Array<Value> fields;
        TupleValue tuple;
        for (Index i = 0; i < instr.alloc_tuple.num_fields; ++i) {
          tuple->fields.push_back(ReadRegister(instr.alloc_tuple.fields[i]));
        }
        WriteRegister(instr.dst, tuple);
        pc_++;
        goto main_loop;
      }
      case Opcode::AllocClosure: {
        std::vector<Value> free_vars;
        for (Index i = 0; i < instr.alloc_closure.num_free_vars; i++) {
          free_vars.push_back(ReadRegister(instr.alloc_closure.free_vars[i]));
        }
        WriteRegister(instr.dst, VMClosureValue::make(instr.alloc_closure.func_index, free_vars));
        pc_++;
        goto main_loop;
      }
      case Opcode::AllocStorage: {
        auto size = LoadScalarInt(instr.alloc_storage.allocation_size);
        auto alignment = instr.alloc_storage.alignment;

        DLOG(INFO) << "AllocStorage: allocation_size=" << size << " alignment=" << alignment
                   << " dtype_hint="
                   << tvm::runtime::DLDataType2String(instr.alloc_storage.dtype_hint);

        auto ctx = GetContext(instr.alloc_storage.device_type, instr.alloc_storage.device_id);
        auto storage = make_storage(size, alignment, instr.alloc_storage.dtype_hint, ctx);
        WriteRegister(instr.dst, storage);
        pc_++;
        goto main_loop;
      }
      case Opcode::Ret: {
        // If we have hit the point from which we started
        // running, we should return to the caller breaking
        // the dispatch loop.
        return_register_ = ReadRegister(instr.result);
        auto caller_return_register = frames_.back().caller_return_register;

        if (PopFrame() == frame_start) {
          return;
          // Otherwise we are just returning from a local call.
        } else {
          WriteRegister(caller_return_register, return_register_);
          goto main_loop;
        }
      }
      case Opcode::InvokeJit: {
        static auto fschema = Op::GetAttrMap<op::FMNMSchema>("FMNMSchema");
        auto call_values = CallValues::make();
        Index num_inputs = instr.invoke_jit.arity - instr.invoke_jit.output_size;
        auto op = Downcast<OpValue>(ReadRegister(instr.invoke_jit.op_reg));
        call_values->callee = op;
        Array<Value> args;
        for (Index i = 0; i < num_inputs; i++) {
          args.push_back(ReadRegister(instr.invoke_jit.args[i]));
        }
        call_values->args = fschema[op->op](args);
        call_values->ctx = ctxs_[0];
        if (instr.invoke_jit.output_size == 1) {
          call_values->out = ReadRegister(instr.invoke_jit.args[num_inputs]);
        } else {
          Array<Value> outs;
          for (Index i = num_inputs; i < instr.invoke_jit.arity; i++) {
            outs.push_back(ReadRegister(instr.invoke_jit.args[i]));
          }
          call_values->out = TupleValue::make(outs);
        }
        std::unique_ptr<OpEnv> op_env = OpDispatch::Dispatch(call_values);
        if (op_env != nullptr) {
          // TODO(vinx13): request stream
          std::shared_ptr<Requests> requests = op_env->GetRequests();
          for (size_t i = 0; i < requests->workspace.size(); i++) {
            Requests::WorkspaceRequest& entry = requests->workspace[i];
            auto buf = memory_pool::Memory::Alloc(entry.ctx, entry.nbytes);
            *entry.dest = buf->data;
          }
          op_env->Execute(call_values);
        } else {
          LOG(FATAL) << "ValueError: Cannot dispatch " << op->op->name << "@"
                     << call_values->ctx.c_str();
          throw;
        }
        pc_++;
        goto main_loop;
      }
    }
  }
}

Context VirtualMachine::GetContext(DevType device_type, int device_id) {
  for (auto& ctx : ctxs_) {
    if (ctx.device_type == device_type && ctx.device_id == device_id) {
      return ctx;
    }
  }
  LOG(FATAL) << "Context with device_type=" << device_type << " device_id=" << device_id
             << " not found";
  return Context();
}

tvm::runtime::Module CreateVirtualMachine(const Executable* exec) {
  auto vm = make_object<VirtualMachine>();
  vm->LoadExecutable(exec);
  return tvm::runtime::Module(vm);
}

MNM_REGISTER_GLOBAL("mnm.vm.VirtualMachine").set_body([](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
  tvm::runtime::Module mod = args[0];
  const auto* exec = dynamic_cast<Executable*>(mod.operator->());
  CHECK(exec) << "The virtual machine executable has not been defined yet.";
  *rv = CreateVirtualMachine(exec);
});

}  // namespace vm
}  // namespace executor
}  // namespace mnm
