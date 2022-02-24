/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/impl/vm/vm_debugger.cc
 * \brief The implementation for RAF virtual machine debugger.
 */
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "tvm/relay/transform.h"
#include "raf/device_api.h"
#include "raf/memory_pool.h"
#include "./vm_debugger.h"

namespace raf {
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

PackedFunc VMDebugger::GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) {
  if (name == "get_interm_tensors") {
    return PackedFunc([sptr_to_self, this](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      ICHECK_EQ(args.size(), 0U);
      CHECK_EQ(op_inputs_.size(), op_names_.size());
      CHECK_EQ(op_outputs_.size(), op_names_.size());
      Map<String, ObjectRef> res{
          {"names", op_names_}, {"inputs", op_inputs_}, {"outputs", op_outputs_}};
      *rv = res;
    });
  } else if (name == "run") {
    return PackedFunc([sptr_to_self, this](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      ICHECK_EQ(args.size(), 1U);
      VMContext ctx = args[0];
      *rv = Run(ctx);
    });
  } else if (name == "reset") {
    return PackedFunc([sptr_to_self, this](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
      op_invokes_.clear();
      op_shapes_.clear();
      op_names_.clear();
      op_inputs_.clear();
      op_outputs_.clear();
      for (auto op_env_cache : op_env_cache_) {
        op_env_cache->Clear();
      }
    });
  } else {
    return VirtualMachine::GetFunction(name, sptr_to_self);
  }
}

void VMDebugger::HandleInvokeJit(VMContext& ctx, const Instruction& instr) {
  OpEnvPtr op_env;
  std::vector<Value> inputs;
  Value output;
  std::string op_env_cache_key;

  std::tie(op_env, inputs, output, op_env_cache_key) = PrepareOpEnv(ctx, instr);
  op_env->Execute(inputs, output);
  ctx->pc++;

  if (op_invokes_.find(op_env.get()) == op_invokes_.end()) {
    op_invokes_[op_env.get()] = 0;

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
    op_shapes_[op_env.get()] = ss.str();
  }
  op_invokes_[op_env.get()]++;

  // cache interm tensors
  Array<Value> input;
  for (const auto& v : inputs) {
    input.push_back(CopyTo(v, host_device_));
  }
  op_inputs_.push_back(input);
  op_outputs_.push_back(CopyTo(output, host_device_));
  op_names_.push_back(op_env->name());
}

tvm::runtime::Module CreateVMDebugger(const Executable* exec) {
  auto vm = make_object<VMDebugger>();
  vm->LoadExecutable(exec);
  return tvm::runtime::Module(vm);
}

RAF_REGISTER_GLOBAL("raf.vm.VMDebugger").set_body([](tvm::TVMArgs args, tvm::TVMRetValue* rv) {
  tvm::runtime::Module mod = args[0];
  const auto* exec = dynamic_cast<Executable*>(mod.operator->());
  CHECK(exec) << "The virtual machine executable has not been defined yet.";
  *rv = CreateVMDebugger(exec);
});

}  // namespace vm
}  // namespace executor
}  // namespace raf
