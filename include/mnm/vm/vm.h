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
#include "mnm/ir.h"
#include "mnm/registry.h"
#include "mnm/value.h"
#include "mnm/vm/bytecode.h"
#include "mnm/vm/executable.h"
#include "mnm/vm/value.h"

namespace mnm {
namespace executor {
namespace vm {

using namespace mnm::ir;
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
  /*! \brief The return program counter. */
  Index pc;
  /*! \brief The index into the function table, points to the caller. */
  Index func_index;
  /*! \brief The number of arguments. */
  Index args;
  /*! \brief A pointer into the caller function's instructions. */
  const Instruction* code;
  /*! \brief Statically allocated space for objects */
  std::vector<Value> register_file;
  /*! \brief Register in caller's frame to put return value */
  RegName caller_return_register;

  VMFrame(Index pc, Index func_index, Index args, const Instruction* code, Index register_file_size)
      : pc(pc),
        func_index(func_index),
        args(args),
        code(code),
        register_file(register_file_size),
        caller_return_register(0) {
  }
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

  VirtualMachine() : frames_(), func_index_(0), code_(nullptr), pc_(0), exec_(nullptr) {
  }

  /*!
   * \brief load the executable for the virtual machine.
   * \param exec The executable.
   */
  virtual void LoadExecutable(const Executable* exec);

 protected:
  /*! \brief The virtual machine's packed function table. */
  std::vector<PackedFunc> packed_funcs_;
  /*! \brief The current stack of call frames. */
  std::vector<VMFrame> frames_;
  /*! \brief The fuction table index of the current function. */
  Index func_index_;
  /*! \brief The current pointer to the code section. */
  const Instruction* code_;
  /*! \brief The virtual machine PC. */
  Index pc_;
  /*! \brief The special return register. */
  Value return_register_;
  /*! \brief The executable the VM will operate on. */
  const Executable* exec_;
  /*! \brief The function name to inputs mapping. */
  std::unordered_map<std::string, std::vector<Value>> inputs_;
  /*! \brief The set of TVM contexts the VM is currently executing on. */
  std::vector<Context> ctxs_;

  /*! \brief Push a call frame on to the call stack. */
  void PushFrame(Index arg_count, Index ret_pc, const VMFunction& vm_func);

  /*!
   * \brief Pop a frame off the call stack.
   * \return The number of frames left.
   */
  Index PopFrame();

  /*!
   * \brief Write to a VM register.
   * \param reg The register to write to.
   * \param obj The object to write to.
   */
  inline void WriteRegister(RegName reg, const Value& obj);

  /*!
   * \brief Read a VM register.
   * \param reg The register to read from.
   * \return The read object.
   */
  inline Value ReadRegister(RegName reg) const;

  /*!
   * \brief Read a VM register and cast it to int32_t
   * \param reg The register to read from.
   * \return The read scalar.
   */
  int64_t LoadScalarInt(RegName reg) const;

  /*!
   * \brief Invoke a VM function.
   * \param func The function.
   * \param args The arguments to the function.
   * \return The value representing the result.
   */
  Value Invoke(const VMFunction& func, const std::vector<Value>& args);

  // TODO(@jroesch): I really would like this to be a global variable.
  /*!
   * \brief Invoke a VM function by name.
   * \param name The function's name.
   * \param args The arguments to the function.
   * \return The value representing the result.
   */
  Value Invoke(const std::string& name, const std::vector<Value>& args);

  /*!
   * \brief Initialize the virtual machine for a set of contexts.
   * \param contexts The set of TVM contexts.
   */
  void Init(const std::vector<Context>& contexts);

  /*! \brief Run VM dispatch loop. */
  void RunLoop();

  /*! \brief Get device context for params. */
  Context GetParamsContext() const;

  /*! \brief Get context by device type and id. */
  Context GetContext(DevType device_type, int device_id);

 private:
  /*!
   * \brief Invoke a global setting up the VM state to execute.
   *
   * This does not begin execution of the VM.
   */
  void InvokeGlobal(const VMFunction& func, const std::vector<Value>& args);

  /*!
   * \brief The constant pool for runtime. It caches the device dependent
   * object to avoid rellocation of constants during inference.
   */
  std::vector<Value> const_pool_;
};

}  // namespace vm
}  // namespace executor
}  // namespace mnm
