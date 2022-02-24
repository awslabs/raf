/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file include/raf/vm/executable.h
 * \brief The RAF virtual machine executable.
 */
#pragma once

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "raf/ir.h"
#include "raf/registry.h"
#include "raf/value.h"

namespace raf {
namespace executor {
namespace vm {

using namespace raf::ir;
using raf::registry::PackedFunc;

struct VMFunction;

/*!
 * \brief The executable emitted by the VM compiler.
 *
 * The executable contains information (e.g. data in different memory regions)
 * to run in a virtual machine.
 *
 *  - Global section, containing all globals.
 *  - Constant section, storing the constant pool.
 *  - Primitive name section, containing the function name of the primitive ops
 *  used by the virtual machine.
 *  - Code section, handling the VM functions and bytecode.
 */
class Executable : public tvm::runtime::ModuleNode {
 public:
  /*!
   * \brief Get a PackedFunc from an executable module.
   *
   * \param name the name of the function.
   * \param sptr_to_self The shared_ptr that points to this module node.
   *
   * \return PackedFunc or nullptr when it is not available.
   */
  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final;

  /*!
   * \brief Serialize the executable into global section, constant section, and
   * code section.
   *
   * \return The binary representation of the VM.
   */
  TVMByteArray Save();

  /*!
   * \brief Load the saved VM executable.
   *
   * \param code The bytecode in string.
   * \param lib The compiled runtime library.
   *
   * \return exe The constructed executable.
   */
  static tvm::runtime::Module Load(const std::string& code, const tvm::runtime::Module lib);

  /*!
   * \brief Get the serialized form of the `functions`. This is
   * essentially bytecode serialization.
   *
   * \return The serialized vm bytecode.
   *
   * \note The bytecode is in the following format:
   *   func_name reg_file_size num_instructions
   *   param1 param2 ... paramM
   *   instruction1
   *   instruction2
   *   ...
   *   instructionN
   *
   * Each instruction is printed in the following format:
   *   opcode num_fields field1 ... fieldX # The text format.
   *
   * Serializing an `Instruction` requires us to deal with the bytecode. Each line
   * of the instructions could be serialized as the following format:
   *   hash, opcode, f1, f2, ..., fX, field with variable length
   *   1. hash: the hash of the instruction. This number will be used to help us
   * validate if an instruction is well-formed during deserialization.
   *   2. opcode: the opcode code of the instruction.
   *   3. f1, f2, ..., fX. These fields together represent the fixed fields in
   * an instruction, e.g., `from` and `dst` fields of a `Move` instruction. For
   * example, `DLDataType` will be unpacked into three fields (code, bits, lanes).
   *   4. The rest of the line indicates the field with variable length, e.g.,
   * the shape of a tensor, the args used by an `InvokPacked` instruction, etc.

   * The field starting from # is only used for debugging. The serialized code
   * doesn't contain it, therefore the deserializer doens't need to handle it.
   */
  std::string GetBytecode() const;

  /*!
   * \brief Print the detailed statistics of the given code, i.e. number of
   * globls and constants, etc.
   */
  std::string Stats() const;

  /*!
   * \brief Get the `lib` module in an executable. Users have the flexibility to call
   * `export_library` from the frontend to save the library to disk.
   *
   * \return The runtime module that contains the hardwre dependent code.
   */
  tvm::runtime::Module GetLib() const {
    return lib;
  }

  /*!
   * \brief Get the arity of the VM Fucntion.
   * \param func Function name.
   * \return The number of parameters.
   */
  int GetFunctionArity(std::string func) const;

  /*!
   * \brief Get the parameter name given the function name and parameter index.
   * \param func Function name.
   * \param index Parameter index.
   * \return The parameter name.
   */
  std::string GetFunctionParameterName(std::string func, uint32_t index) const;

  virtual ~Executable() {
  }

  const char* type_key() const final {
    return "raf.executor.vm.VMExecutable";
  }

  /*! \brief The runtime module/library that contains both the host and also the device
   * code when executing on non-CPU devices. */
  tvm::runtime::Module lib;
  /*! \brief The global constant pool. */
  std::vector<Value> constants;
  /*! \brief A map from globals (as strings) to their index in the function map. */
  std::unordered_map<std::string, Index> global_map;
  /*! \brief A mapping from the packed function (as string) to the index that
   * corresponds to the position of the `packed_funcs` list in a `VirtualMachine` object.
   */
  std::unordered_map<std::string, Index> primitive_map;
  /*! \brief The virtual machine's function table. */
  std::vector<VMFunction> functions;

 private:
  /*!
   * \brief Save the globals.
   *
   * \param strm The input stream.
   */
  void SaveGlobalSection(dmlc::Stream* strm);

  /*!
   * \brief Save the constant pool.
   *
   * \param strm The input stream.
   */
  void SaveConstantSection(dmlc::Stream* strm);

  /*!
   * \brief Save primitive op names.
   *
   *  \param strm The input stream.
   */
  void SavePrimitiveOpNames(dmlc::Stream* strm);

  /*!
   * \brief Save the vm functions.
   *
   * \param strm The input stream.
   */
  void SaveCodeSection(dmlc::Stream* strm);

  /*!
   * \brief Load the globals.
   *
   * \param strm The input stream.
   */
  void LoadGlobalSection(dmlc::Stream* strm);

  /*!
   * \brief Load the constant pool.
   *
   * \param strm The input stream.
   */
  void LoadConstantSection(dmlc::Stream* strm);

  /*!
   * \brief Load primitive op names.
   *
   * \param strm The input stream.
   */
  void LoadPrimitiveOpNames(dmlc::Stream* strm);

  /*!
   * \brief Load the vm functions.
   *
   * \param strm The input stream.
   */
  void LoadCodeSection(dmlc::Stream* strm);

  /*! \brief The serialized bytecode. */
  std::string code_;
};

}  // namespace vm
}  // namespace executor
}  // namespace raf
