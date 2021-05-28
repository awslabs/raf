/*!
 * Copyright (c) 2020 by Contributors
 * \file include/mnm/vm/vm.h
 * \brief A virtual machine for executing Meta programs.
 */
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "bytecode.h"
#include "executable.h"
#include "mnm/ir.h"
#include "mnm/registry.h"
#include "mnm/value.h"
#include "mnm/op.h"
#include "mnm/op_utils.h"
#include "mnm/vm/bytecode.h"
#include "mnm/vm/executable.h"
#include "mnm/vm/value.h"

namespace mnm {
namespace executor {
namespace vm {

using namespace mnm::ir;
using namespace mnm::op;
using namespace mnm::value;
using mnm::registry::PackedFunc;

/*! \brief Magic number for NDArray list file  */
constexpr uint64_t kTVMNDArrayListMagic = 0xF7E58D4F05049CB7;

/*!
 * \brief A representation of a Relay function in the VM.
 *
 * Contains metadata about the compiled function, as
 * well as the compiled VM instructions.
 */
struct VMFunction {
  /*! \brief The function's name. */
  std::string name;
  /*! \brief The function parameter names. */
  std::vector<std::string> params;
  /*! \brief The instructions representing the function. */
  std::vector<Instruction> instructions;
  /*! \brief The size of the frame for this function */
  Index register_file_size;

  VMFunction(const std::string& name, std::vector<std::string> params,
             std::vector<Instruction> instructions, Index register_file_size)
      : name(name),
        params(std::move(params)),
        instructions(std::move(instructions)),
        register_file_size(register_file_size) {
  }

  VMFunction() {
  }

  friend std::ostream& operator<<(std::ostream& os, const VMFunction&);
};

/*!
 * \brief A representation of a stack frame.
 *
 * A stack frame is a record containing the information needed
 * to restore the caller's virtual machine state after returning
 * from a function call.
 */
struct VMFrame {
  /*! \brief The caller function index in the function table. */
  Index caller_func_index;
  /*! \brief The return program counter in caller. */
  Index caller_return_pc;
  /*! \brief Register in caller's frame to put return value. */
  RegName caller_return_register;
  /*! \brief The number of arguments. */
  Index num_args;
  /*! \brief Statically allocated space for objects. */
  std::vector<Value> register_file;
  /*! \brief Indicate whether each register is constant. */
  std::vector<bool> is_const;

  VMFrame(Index caller_func_index, Index caller_pc, RegName caller_ret_reg, Index num_args,
          Index register_file_size)
      : caller_func_index(caller_func_index),
        caller_return_pc(caller_pc),
        caller_return_register(caller_ret_reg),
        num_args(num_args),
        register_file(register_file_size),
        is_const(register_file_size, false) {
  }
};

/*!
 * \brief VMContextObj holds the runtime data for an execution in the VM.
 */
class VMContextObj : public ValueObj {
 public:
  /*! \brief The current stack of call frames. */
  std::vector<VMFrame> frames;
  /*! \brief The fuction table index of the current function. */
  Index func_index{-1};
  /*! \brief The virtual machine PC. */
  Index pc{0};
  /*! \brief The current pointer to the code section. */
  const Instruction* code{nullptr};
  /*! \brief The final return register. */
  Value return_register;
  /*! \brief The entry function index to invoke. */
  Index entry_func_index;
  /*! \brief The input arguments. */
  std::vector<Value> inputs;
  /*! \brief The pointer to the executable. */
  const Executable* exec;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("func_index", &func_index);
    v->Visit("pc", &pc);
    v->Visit("return_register", &return_register);
    v->Visit("entry_func_index", &entry_func_index);
  }
  static constexpr const uint32_t _type_index = ir::TypeIndex::kDynamic;
  static constexpr const char* _type_key = "mnm.vm.VMContext";
  MNM_BASE_OBJECT(VMContextObj, ValueObj);
};

/*!
 * \brief VMContext is the wrapper for the VMContextObj and provides additional
 *   APIs to access and update the runtime context.
 *
 * The APIs in the VMContext are NOT thread safe.
 */
class VMContext : public Value {
 public:
  static VMContext make(const Executable* exec);
  /*!
   * \brief Read a VM register.
   * \param reg The register to read from.
   * \return The read object.
   */
  inline Value ReadRegister(Index reg) const;
  /*!
   * \brief Write to a VM register.
   * \param reg The register to write to.
   * \param val The value to write.
   */
  inline void WriteRegister(Index reg, const Value& val);
  /*!
   * \brief Read a VM register and cast it to int64_t
   * \param reg The register to read from.
   * \return The read scalar.
   */
  inline int64_t LoadScalarInt(Index reg) const;
  /*!
   * \brief Check if a VM register is constant.
   * \param reg The register to check.
   * \return Whether the register is constant.
   */
  inline bool IsConst(Index reg) const;
  /*!
   * \brief Push a call frame on to the call stack.
   * \param func_index The index of the VM function to invoke.
   * \param args The arguments to the function.
   * \param ret_reg The return register to write back in the caller.
   */
  inline void PushFrame(Index func_index, const std::vector<Value>& args, RegName ret_reg);
  /*!
   * \brief Pop a frame off the call stack.
   * \return The number of frnames left.
   */
  inline Index PopFrame();

  MNM_OBJECT_REF(VMContext, Value, VMContextObj);
};

using OpEnvCache = MetaCache<std::shared_ptr<OpEnv>>;

/*! \brief The OpEnv cache for a VM function. */
class VMFuncOpEnvCache {
 public:
  /*!
   * \brief Get the OpEnv cache for a given instruction.
   * \param pc The program counter
   * \return The OpEnv cache.
   */
  std::shared_ptr<OpEnvCache> Get(Index pc);

  /*!
   * \brief Clear the OpEnv cache.
   */
  void Clear();

 private:
  /*! \brief Cache map from instruction index to OpEnv cache. */
  std::unordered_map<Index, std::shared_ptr<OpEnvCache>> cache_map_;
  /*! \brief The mutex for the cache_map_. */
  std::mutex mu_;
};

/*!
 * \brief The virtual machine.
 *
 * The virtual machine contains all the current execution state,
 * as well as the executable.
 *
 * The goal is to have a single self-contained object,
 * enabling one to easily pass around VMs, execute them on
 * multiple threads, or serialize them to disk or over the
 * wire.
 */
class VirtualMachine : public tvm::runtime::ModuleNode {
 public:
  /*!
   * \brief Get a PackedFunc from module.
   *
   *  The PackedFunc may not be fully initialized,
   *  there might still be first time running overhead when
   *  executing the function on certain devices.
   *  For benchmarking, use prepare to eliminate
   *
   * \param name the name of the function.
   * \param sptr_to_self The shared_ptr that points to this module node.
   *
   * \return PackedFunc(nullptr) when it is not available.
   *
   * \note The function will always remain valid.
   *   If the function needs resource from the module(e.g. late linking),
   *   it should capture sptr_to_self.
   */
  virtual PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self);

  virtual ~VirtualMachine() {
  }

  const char* type_key() const final {
    return "VirtualMachine";
  }

  VirtualMachine(bool enable_cuda_graph) : exec_(nullptr), enable_cuda_graph_(enable_cuda_graph) {
#ifndef MNM_USE_CUDA
    if (enable_cuda_graph) {
      LOG(WARNING) << "Because CUDA is not enabled in Meta, CUDA graph will be disabled in the VM.";
      enable_cuda_graph_ = false;
    }
#endif
    if (enable_cuda_graph_) {
      LOG(WARNING) << "Concurrent execution is not supported for VM in CUDA graph mode.";
    }
  }

  /*!
   * \brief load the executable for the virtual machine.
   * \param exec The executable.
   */
  virtual void LoadExecutable(const Executable* exec);
  /*!
   * \brief Initialize the virtual machine for a set of devices.
   * \param devices The set of devices.
   */
  void SetDevices(const std::vector<Device>& devices);
  /*!
   * \brief Prepare a VM runtime context.
   * \param func_name The entry function name.
   * \param inputs The inputs to the function.
   * \return The VM context.
   */
  VMContext PrepareVMContext(const std::string& func_name, const std::vector<Value>& inputs);
  /*!
   * \brief Run the virtual machine.
   * \param ctx The runtime context.
   * \return The return value.
   */
  Value Run(VMContext ctx);

 protected:
  /*! \brief The virtual machine's packed function table. */
  std::vector<PackedFunc> packed_funcs_;
  /*! \brief The executable the VM will operate on. */
  const Executable* exec_;
  /*! \brief The set of devices the VM is currently executing on. */
  std::vector<Device> devices_;

  /*! \brief Run VM dispatch loop. */
  void RunLoop(VMContext ctx);
  /*! \brief Get device context for params. */
  Device GetParamsDevice() const;

  /*! \brief Prepare an OpEnv with its inputs and output */
  virtual std::tuple<std::shared_ptr<OpEnv>, std::vector<Value>, Value> PrepareOpEnv(
      const VMContext& ctx, const Instruction& instr);

  /*! \brief Execute OpEnv */
  virtual void ExecuteOpEnv(OpEnv*, const std::vector<value::Value>& inputs, value::Value output);

  /*! \brief Run InferType Instruction */
  virtual void RunInferType(VMContext& ctx, const Instruction& instr);

 protected:
  /*!
   * \brief The constant pool for runtime. It caches the device dependent
   * object to avoid rellocation of constants during inference.
   */
  std::vector<Value> const_pool_;
  /*!
   * \brief OpEnv cache. Each element in the vector stores the cache for the
   * corresponding VM function. It's a map from pc to the OpEnv cache.
   */
  std::vector<std::shared_ptr<VMFuncOpEnvCache>> op_env_cache_;

#ifdef MNM_USE_CUDA
  /*!
   * \brief A class to store, cache and execute CUDA Graph
   *
   * Cached CUDA Graph is stored in this class, as well as
   * stream for capturing.
   */
  class CudaGraphImpl;
  /*! \brief A pointer into the CUDA Graph instance. */
  CudaGraphImpl* cuda_graph_impl_ = nullptr;
  /*! \brief The context associated with the captured CUDA graph. */
  VMContext cuda_graph_ctx_;
  /*! \brief Indicate whether the CUDA graph is currently in use by a context. */
  bool cuda_graph_occupied_ = false;
  /*! \brief The mutex to access CUDA graph related fields. */
  std::mutex cuda_graph_mutex_;
#endif

  /*! \brief Indicates whether CUDA Graph is enabled when VM is initialized. */
  bool enable_cuda_graph_ = false;
};

}  // namespace vm
}  // namespace executor
}  // namespace mnm
