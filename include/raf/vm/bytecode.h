/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file include/raf/vm/bytecode.h
 * \brief The bytecode for Relay virtual machine.
 */
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "raf/ir_ext.h"
#include "raf/registry.h"
#include "raf/value.h"

namespace raf {
namespace executor {
namespace vm {

using namespace raf::ir;
using raf::registry::PackedFunc;
using raf::value::Value;

/*! \brief A register name. */
using RegName = int64_t;

/*! \brief An alias for the integer type used ubiquitously
 * in the VM.
 */
using Index = int64_t;

/*! \brief An enumeration of Relay's opcodes.
 *
 * The opcode is used to implement instruction
 * as a tagged union.
 */
enum class Opcode {
  //  Basic instructions
  Move = 0U,
  Ret = 1U,
  Fatal = 2U,
  LoadConst = 3U,
  LoadConsti = 4U,
  GetField = 5U,
  // TODO(@icemelon9): Current don't support ADT object
  // GetTag = 6U,

  // Control instructions
  If = 10U,
  Goto = 11U,

  // Memory instructions
  AllocStorage = 20U,
  AllocTensor = 21U,
  AllocTensorReg = 22U,
  AllocTuple = 23U,
  AllocClosure = 24U,
  // TODO(@icemelon9): Current don't support ADT object
  // AllocADT = 25U,
  SetShape = 26U,
  Free = 27U,

  // Invoke instructions
  InvokeFunc = 30U,
  InvokeClosure = 31U,
  InvokePacked = 32U,
  InvokeJit = 33U,
  InferType = 34U,

  // Cuda stream instructions
  CudaSetStream = 40U,
  CudaAddEvent = 41U,
  CudaWaitEvent = 42U,
  CudaStreamBarrier = 43U,
};

/*! \brief A single virtual machine instruction.
 *
 * The representation of the instruction is as
 * a tagged union.
 *
 * The first field represents which instruction,
 * and by extension which field of the union
 * is active.
 */
struct Instruction {
  /*! \brief The instruction opcode. */
  Opcode op;

  /*! \brief The destination register. */
  RegName dst;

  union {
    // Basic instructions
    struct /* Move Operands */ {
      /*! \brief The source register for a move operation. */
      RegName from;
    };
    struct /* Return Operands */ {
      /*! \brief The register to return. */
      RegName result;
    };
    struct /* LoadConst Operands */ {
      /* \brief The index into the constant pool. */
      Index const_index;
    };
    struct /* LoadConsti Operands */ {
      /* \brief The index into the constant pool. */
      Index val;
    } load_consti;
    struct /* GetField Operands */ {
      /*! \brief The register to read from. */
      RegName object;
      /*! \brief The field to read out. */
      Index field_index;
    } get_field;

    // Control flow instructions
    struct /* If Operands */ {
      /*! \brief The register containing the test value. */
      RegName test;
      /*! \brief The register containing the target value. */
      RegName target;
      /*! \brief The program counter offset for the true branch. */
      Index true_offset;
      /*! \brief The program counter offset for the false branch. */
      Index false_offset;
    } if_op;
    struct /* Jump Operands */ {
      /*! \brief The jump offset. */
      Index pc_offset;
    };

    // Memory instructions
    struct /* AllocStorage Operands */ {
      /*! \brief The size of the allocation. */
      RegName allocation_size;
      /*! \brief The alignment of the allocation. */
      Index alignment;
      /*! \brief The hint of the dtype. */
      DLDataType dtype_hint;
      /*! \brief The allocated device type. */
      DevType device_type;
      /*! \brief The allocated device ID. */
      Index device_id;
      /*! \brief Allocate storage alloc_async if available. */
      bool alloc_async;
    } alloc_storage;
    struct /* AllocTensor Operands */ {
      /*! \brief The storage to allocate from. */
      RegName storage;
      /*! \brief The offset into the storage to allocate from. */
      Index offset;
      /*! \brief The number of dimensions. */
      uint32_t ndim;
      /*! \brief The shape of tensor. */
      int64_t* shape;
      /*! \brief The datatype of tensor to be allocated. */
      DLDataType dtype;
      /*! \brief Whether the tensor should own the storage memory. */
      bool own;
    } alloc_tensor;
    struct /* Free Operands */ {
      /*! \brief The memory to be freed. It can be a tensor or a storage. */
      RegName memory;
    } free;
    struct /* AllocTensorReg Operands */ {
      /*! \brief The storage to allocate from. */
      RegName storage;
      /*! \brief The offset into the storage to allocate from. */
      Index offset;
      /*! \brief The register to read the shape out of. */
      RegName shape_register;
      /*! \brief The datatype of tensor to be allocated. */
      DLDataType dtype;
      /*! \brief Whether the tensor should own the storage memory. */
      bool own;
    } alloc_tensor_reg;
    struct /* AllocClosure Operands */ {
      /*! \brief The index into the function table. */
      Index func_index;
      /*! \brief The number of free variables to capture. */
      Index num_free_vars;
      /*! \brief The free variables as an array. */
      RegName* free_vars;
    } alloc_closure;
    struct /* AllocTuple Operands */ {
      /*! \brief The number of fields to store in the tuple. */
      Index num_fields;
      /*! \brief The fields in the tuple. */
      RegName* fields;
    } alloc_tuple;
    struct /* SetShape Operands */ {
      /*! \brief The register containing the data. */
      RegName data;
      /*! \brief The register containing the shape. */
      RegName shape;
    } set_shape;

    struct /* InvokeFunc Operands */ {
      /*! \brief The function to call. */
      Index func_index;
      /*! \brief The number of arguments to the function. */
      Index num_args;
      /*! \brief The registers containing the arguments. */
      RegName* args;
    } invoke_func;
    struct /* InvokeClosure Operands */ {
      /*! \brief The register containing the closure. */
      RegName closure;
      /*! \brief The number of arguments to the closure. */
      Index num_args;
      /*! \brief The closure arguments as an array. */
      RegName* args;
    } invoke_closure;
    struct /* InvokePacked Operands */ {
      /*! \brief The index into the packed function table. */
      Index packed_index;
      /*! \brief The arity of the packed function. */
      Index arity;
      /*! \brief The number of outputs produced by the packed function. */
      Index output_size;
      /*! \brief The arguments to pass to the packed function. */
      RegName* args;
    } invoke_packed;
    struct /* InvokeJit Operands */ {
      /*! \brief The register containing the OpValue to invoke. */
      RegName op_reg;
      /*! \brief The arity of the packed function. */
      Index arity;
      /*! \brief The number of outputs produced by the packed function. */
      Index output_size;
      /*! \brief The arguments to pass to the packed function. */
      RegName* args;
    } invoke_jit;
    struct /* InferType Operands */ {
      /*! \brief The register containing the OpValue to invoke OpType. */
      RegName op_reg;
      /*! \brief The number of arguments to the origin call */
      Index num_args;
      /*! \brief The registers containing the arguments. */
      RegName* args;
    } infer_type;
    struct /* CudaSetStream Operands */ {
      /*! \brief The id of the cuda device that we want to set the stream */
      Index device_id;
      /*! \brief The id of the target stream */
      Index stream_id;
    } cuda_set_stream;
    struct /* CudaAddEvent and CudaWaitEvent Operands */ {
      /*! \brief The id of the event need to add or wait on current device */
      Index event_id;
      /*! \brief The id of the stream need to add or wait on current device */
      Index stream_id;
    } cuda_event;
  };

  /*!
   * \brief Construct a return instruction.
   * \param return_reg The register containing the return value.
   * \return The return instruction.
   */
  static Instruction Ret(RegName return_reg);
  /*!
   * \brief Construct a fatal instruction.
   * \return The fatal instruction.
   */
  static Instruction Fatal();
  /*!
   * \brief Construct a invoke packed instruction.
   * \param packed_index The index of the packed function.
   * \param arity The arity of the function.
   * \param output_size The number of outputs of the packed function.
   * \param args The argument registers.
   * \return The invoke packed instruction.
   */
  static Instruction InvokePacked(Index packed_index, Index arity, Index output_size,
                                  const std::vector<RegName>& args);
  /*!
   * \brief Construct an allocate tensor instruction with constant shape.
   * \param storage The storage to allocate out of.
   * \param offset The offset to allocate at.
   * \param shape The shape of the tensor.
   * \param dtype The dtype of the tensor.
   * \param dst The destination register.
   * \param own Whether to own the storage memory.
   * \return The allocate tensor instruction.
   */
  static Instruction AllocTensor(RegName storage, Index offset, const std::vector<int64_t>& shape,
                                 DLDataType dtype, RegName dst, bool own = true);
  /*!
   * \brief Construct an allocate tensor instruction with register.
   * \param storage The storage to allocate out of.
   * \param offset The offset to allocate at.
   * \param shape_register The register containing the shape.
   * \param dtype The dtype of the tensor.
   * \param dst The destination register.
   * \param own Whether to own the storage memory.
   * \return The allocate tensor instruction.
   */
  static Instruction AllocTensorReg(RegName storage, Index offset, RegName shape_register,
                                    DLDataType dtype, RegName dst, bool own = true);
  /*!
   * \brief Construct an allocate tuple instruction.
   * \param num_fields The number of fields for the datatype.
   * \param fields The registers containing the fields.
   * \param dst The register name of the destination.
   * \return The allocate tuple instruction.
   */
  static Instruction AllocTuple(const std::vector<RegName>& fields, RegName dst);
  /*!
   * \brief Construct an allocate closure instruction.
   * \param func_index The index of the function table.
   * \param free_vars The registers of the free variables.
   * \param dst The destination register.
   * \return The allocate closure instruction.
   */
  static Instruction AllocClosure(Index func_index, const std::vector<RegName>& free_vars,
                                  RegName dst);
  /*!
   * \brief Construct an set shape instruction.
   * \param data The register containing the data.
   * \param shape The register containing the raw shape.
   * \param dst The destination register.
   * \return The set shape instruction.
   */
  static Instruction SetShape(RegName data, RegName shape, RegName dst);
  /*!
   * \brief Construct a get field instruction.
   * \param object_reg The register containing the object to project from.
   * \param field_index The field to read out of the object.
   * \param dst The destination register.
   * \return The get field instruction.
   */
  static Instruction GetField(RegName object_reg, Index field_index, RegName dst);
  /*!
   * \brief Construct an if instruction.
   * \param test The register containing the test value.
   * \param target The register containing the target value.
   * \param true_branch The offset to the true branch.
   * \param false_branch The offset to the false branch.
   * \return The if instruction.
   */
  static Instruction If(RegName test, RegName target, Index true_branch, Index false_branch);
  /*!
   * \brief Construct a goto instruction.
   * \param pc_offset The offset from the current pc.
   * \return The goto instruction.
   */
  static Instruction Goto(Index pc_offset);
  /*!
   * \brief Construct an invoke instruction.
   * \param func_index The index of the function to invoke.
   * \param args The registers containing the arguments.
   * \param dst The destination register.
   * \return The invoke instruction.
   */
  static Instruction InvokeFunc(Index func_index, const std::vector<RegName>& args, RegName dst);
  /*!
   * \brief Construct an invoke closure instruction.
   * \param closure The register of the closure to invoke.
   * \param args The registers containing the arguments.
   * \param dst The destination register.
   * \return The invoke closure instruction.
   */
  static Instruction InvokeClosure(RegName closure, const std::vector<RegName>& args, RegName dst);
  /*!
   * \brief Construct a load constant instruction.
   * \param const_index The index of the constant.
   * \param dst The destination register.
   * \return The load constant instruction.
   */
  static Instruction LoadConst(Index const_index, RegName dst);
  /*!
   * \brief Construct a load_constanti instruction.
   * \param val The interger constant value.
   * \param dst The destination register.
   * \return The load_constanti instruction.
   */
  static Instruction LoadConsti(Index val, RegName dst);
  /*!
   * \brief Construct a move instruction.
   * \param src The source register.
   * \param dst The destination register.
   * \return The move instruction.
   */
  static Instruction Move(RegName src, RegName dst);

  /*!
   * \brief Allocate a storage block.
   * \param size The size of the allocation.
   * \param alignment The allocation's alignment.
   * \param dtype_hint The data type hint for the allocator.
   * \param device_type The device type.
   * \param device_id The device ID.
   * \param dst The destination to place the storage.
   * \param alloc_async Allocate storage async if available.
   * \return The alloc storage instruction.
   */
  static Instruction AllocStorage(RegName size, RegName alignment, DLDataType dtype_hint,
                                  DevType device_type, Index device_id, RegName dst,
                                  bool alloc_async = true);

  /*!
   * \brief Free a tensor or a storage.
   * \param memory The memory to be freed.
   * \return The free instruction.
   */
  static Instruction Free(RegName memory);

  /*!
   * \brief Construct an invoke JIT operator instruction.
   * \param op_reg The register containing the OpValue to invoke.
   * \param arity The arity of the function.
   * \param output_size The number of outputs of the packed function.
   * \param args The argument registers.
   * \return The invoke JIT operator instruction.
   */
  static Instruction InvokeJit(RegName op_reg, Index arity, Index output_size,
                               const std::vector<RegName>& args);
  /*!
   * \brief Construct an InferType instruction.
   * \param op_reg The register containing the OpValue to invoke OpType.
   * \param args The registers containing the arguments.
   * \param dst The destination register.
   * \return The invoke OpType instruction.
   */
  static Instruction InferType(RegName op_reg, const std::vector<RegName>& args, RegName dst);
  /*!
   * \brief Construct a CudaSetStream instruction.
   * \param device_id The id of device we want to set the stream on.
   * \param stream_id The id of target stream.
   * \return The set stream instruction.
   */
  static Instruction CudaSetStream(Index device_id, Index stream_id);
  /*!
   * \brief Construct a CudaAddEvent instruction.
   * \param event_id The id of event we would use to record.
   * \param stream_id The id of the stream we record on. -1 for current stream.
   * \return The add event instruction.
   */
  static Instruction CudaAddEvent(Index event_id, Index stream_id);
  /*!
   * \brief Construct a CudaWaitEvent instruction.
   * \param event_id The id of event we want to wait for.
   * \param stream_id The id of the stream to wait. -1 for current stream.
   * \return The wait event instruction.
   */
  static Instruction CudaWaitEvent(Index event_id, Index stream_id);

  /*!
   * \brief Construct a CudaStreamBarrier instruction.
   * \return The stream barrier instruction.
   */
  static Instruction CudaStreamBarrier();

  Instruction();
  Instruction(const Instruction& instr);
  Instruction& operator=(const Instruction& instr);
  ~Instruction();

  friend std::ostream& operator<<(std::ostream& os, const Instruction&);
};

}  // namespace vm
}  // namespace executor
}  // namespace raf
