/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/impl/vm/bytecode.cc
 * \brief Byte code of the RAF virtual machine.
 */

#include <dmlc/memory_io.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/object.h>

#include <algorithm>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "raf/vm/vm.h"

namespace raf {
namespace executor {
namespace vm {

Instruction::Instruction() {
}

template <typename T>
static T* Duplicate(T* src, Index size) {
  auto dst = new T[size];
  std::copy(src, src + size, dst);
  return dst;
}

Instruction::Instruction(const Instruction& instr) {
  this->op = instr.op;
  this->dst = instr.dst;

  switch (instr.op) {
    case Opcode::Move:
      this->from = instr.from;
      return;
    case Opcode::Fatal:
      return;
    case Opcode::Ret:
      this->result = instr.result;
      return;
    case Opcode::AllocTensor:
      this->alloc_tensor.storage = instr.alloc_tensor.storage;
      this->alloc_tensor.offset = instr.alloc_tensor.offset;
      this->alloc_tensor.ndim = instr.alloc_tensor.ndim;
      this->alloc_tensor.shape =
          Duplicate<int64_t>(instr.alloc_tensor.shape, instr.alloc_tensor.ndim);
      this->alloc_tensor.dtype = instr.alloc_tensor.dtype;
      this->alloc_tensor.own = instr.alloc_tensor.own;
      return;
    case Opcode::AllocTensorReg:
      this->alloc_tensor_reg = instr.alloc_tensor_reg;
      return;
    case Opcode::AllocTuple:
      this->alloc_tuple.num_fields = instr.alloc_tuple.num_fields;
      this->alloc_tuple.fields =
          Duplicate<RegName>(instr.alloc_tuple.fields, instr.alloc_tuple.num_fields);
      return;
    case Opcode::AllocClosure:
      this->alloc_closure.func_index = instr.alloc_closure.func_index;
      this->alloc_closure.num_free_vars = instr.alloc_closure.num_free_vars;
      this->alloc_closure.free_vars =
          Duplicate<RegName>(instr.alloc_closure.free_vars, instr.alloc_closure.num_free_vars);
      return;
    case Opcode::SetShape:
      this->set_shape.data = instr.set_shape.data;
      this->set_shape.shape = instr.set_shape.shape;
      return;
    case Opcode::InvokePacked:
      this->invoke_packed.packed_index = instr.invoke_packed.packed_index;
      this->invoke_packed.arity = instr.invoke_packed.arity;
      this->invoke_packed.output_size = instr.invoke_packed.output_size;
      this->invoke_packed.args =
          Duplicate<RegName>(instr.invoke_packed.args, instr.invoke_packed.arity);
      return;
    case Opcode::InvokeClosure:
      this->invoke_closure.closure = instr.invoke_closure.closure;
      this->invoke_closure.num_args = instr.invoke_closure.num_args;
      this->invoke_closure.args =
          Duplicate<RegName>(instr.invoke_closure.args, instr.invoke_closure.num_args);
      return;
    case Opcode::InvokeFunc:
      this->invoke_func.func_index = instr.invoke_func.func_index;
      this->invoke_func.num_args = instr.invoke_func.num_args;
      this->invoke_func.args =
          Duplicate<RegName>(instr.invoke_func.args, instr.invoke_func.num_args);
      return;
    case Opcode::If:
      this->if_op = instr.if_op;
      return;
    case Opcode::LoadConst:
      this->const_index = instr.const_index;
      return;
    case Opcode::LoadConsti:
      this->load_consti = instr.load_consti;
      return;
    case Opcode::GetField:
      this->get_field = instr.get_field;
      return;
    case Opcode::Goto:
      this->pc_offset = instr.pc_offset;
      return;
    case Opcode::AllocStorage:
      this->alloc_storage = instr.alloc_storage;
      return;
    case Opcode::Free:
      this->free = instr.free;
      return;
    case Opcode::InvokeJit:
      this->invoke_jit.op_reg = instr.invoke_jit.op_reg;
      this->invoke_jit.arity = instr.invoke_jit.arity;
      this->invoke_jit.output_size = instr.invoke_jit.output_size;
      this->invoke_jit.args = Duplicate<RegName>(instr.invoke_jit.args, instr.invoke_jit.arity);
      return;
    case Opcode::InferType:
      this->infer_type.op_reg = instr.infer_type.op_reg;
      this->infer_type.num_args = instr.infer_type.num_args;
      this->infer_type.args = Duplicate<RegName>(instr.infer_type.args, instr.infer_type.num_args);
      return;
    case Opcode::CudaSetStream:
      this->cuda_set_stream.device_id = instr.cuda_set_stream.device_id;
      this->cuda_set_stream.stream_id = instr.cuda_set_stream.stream_id;
      return;
    case Opcode::CudaAddEvent:
    case Opcode::CudaWaitEvent:
      this->cuda_event.event_id = instr.cuda_event.event_id;
      this->cuda_event.stream_id = instr.cuda_event.stream_id;
      return;
    case Opcode::CudaStreamBarrier:
      return;
    default:
      std::ostringstream out;
      out << "Invalid instruction " << static_cast<int>(instr.op);
      throw std::runtime_error(out.str());
  }
}

template <typename T>
static inline void FreeIf(T* t) {
  if (t != nullptr) {
    delete t;
  }
}

Instruction& Instruction::operator=(const Instruction& instr) {
  this->op = instr.op;
  this->dst = instr.dst;

  switch (instr.op) {
    case Opcode::Move:
      this->from = instr.from;
      return *this;
    case Opcode::Fatal:
      return *this;
    case Opcode::LoadConsti:
      this->load_consti = instr.load_consti;
      return *this;
    case Opcode::Ret:
      this->result = instr.result;
      return *this;
    case Opcode::AllocTensor:
      this->alloc_tensor.storage = instr.alloc_tensor.storage;
      this->alloc_tensor.ndim = instr.alloc_tensor.ndim;
      this->alloc_tensor.shape =
          Duplicate<int64_t>(instr.alloc_tensor.shape, instr.alloc_tensor.ndim);
      this->alloc_tensor.dtype = instr.alloc_tensor.dtype;
      this->alloc_tensor.own = instr.alloc_tensor.own;
      return *this;
    case Opcode::AllocTensorReg:
      this->alloc_tensor_reg.storage = instr.alloc_tensor_reg.storage;
      this->alloc_tensor_reg.shape_register = instr.alloc_tensor_reg.shape_register;
      this->alloc_tensor_reg.dtype = instr.alloc_tensor_reg.dtype;
      this->alloc_tensor_reg.own = instr.alloc_tensor_reg.own;
      return *this;
    case Opcode::AllocTuple:
      this->alloc_tuple.num_fields = instr.alloc_tuple.num_fields;
      FreeIf(this->alloc_tuple.fields);
      this->alloc_tuple.fields =
          Duplicate<RegName>(instr.alloc_tuple.fields, instr.alloc_tuple.num_fields);
      return *this;
    case Opcode::AllocClosure:
      this->alloc_closure.func_index = instr.alloc_closure.func_index;
      this->alloc_closure.num_free_vars = instr.alloc_closure.num_free_vars;
      FreeIf(this->alloc_closure.free_vars);
      this->alloc_closure.free_vars =
          Duplicate<RegName>(instr.alloc_closure.free_vars, instr.alloc_closure.num_free_vars);
      return *this;
    case Opcode::SetShape:
      this->set_shape.data = instr.set_shape.data;
      this->set_shape.shape = instr.set_shape.shape;
      return *this;
    case Opcode::InvokePacked:
      this->invoke_packed.packed_index = instr.invoke_packed.packed_index;
      this->invoke_packed.arity = instr.invoke_packed.arity;
      this->invoke_packed.output_size = instr.invoke_packed.output_size;
      FreeIf(this->invoke_packed.args);
      this->invoke_packed.args =
          Duplicate<RegName>(instr.invoke_packed.args, instr.invoke_packed.arity);
      return *this;
    case Opcode::InvokeClosure:
      this->invoke_closure.closure = instr.invoke_closure.closure;
      this->invoke_closure.num_args = instr.invoke_closure.num_args;
      FreeIf(this->invoke_closure.args);
      this->invoke_closure.args =
          Duplicate<RegName>(instr.invoke_closure.args, instr.invoke_closure.num_args);
      return *this;
    case Opcode::InvokeFunc:
      this->invoke_func.func_index = instr.invoke_func.func_index;
      this->invoke_func.num_args = instr.invoke_func.num_args;
      FreeIf(this->invoke_func.args);
      this->invoke_func.args =
          Duplicate<RegName>(instr.invoke_func.args, instr.invoke_func.num_args);
      return *this;
    case Opcode::InvokeJit:
      this->invoke_jit.op_reg = instr.invoke_jit.op_reg;
      this->invoke_jit.arity = instr.invoke_jit.arity;
      this->invoke_jit.output_size = instr.invoke_jit.output_size;
      FreeIf(this->invoke_jit.args);
      this->invoke_jit.args = Duplicate<RegName>(instr.invoke_jit.args, instr.invoke_jit.arity);
      return *this;
    case Opcode::InferType:
      this->infer_type.op_reg = instr.infer_type.op_reg;
      this->infer_type.num_args = instr.infer_type.num_args;
      FreeIf(this->infer_type.args);
      this->infer_type.args = Duplicate<RegName>(instr.infer_type.args, instr.infer_type.num_args);
      return *this;
    case Opcode::If:
      this->if_op = instr.if_op;
      return *this;
    case Opcode::LoadConst:
      this->const_index = instr.const_index;
      return *this;
    case Opcode::GetField:
      this->get_field = instr.get_field;
      return *this;
    case Opcode::Goto:
      this->pc_offset = instr.pc_offset;
      return *this;
    case Opcode::AllocStorage:
      this->alloc_storage = instr.alloc_storage;
      return *this;
    case Opcode::Free:
      this->free = instr.free;
      return *this;
    case Opcode::CudaSetStream:
      this->cuda_set_stream.device_id = instr.cuda_set_stream.device_id;
      this->cuda_set_stream.stream_id = instr.cuda_set_stream.stream_id;
      return *this;
    case Opcode::CudaAddEvent:
    case Opcode::CudaWaitEvent:
      this->cuda_event.event_id = instr.cuda_event.event_id;
      this->cuda_event.stream_id = instr.cuda_event.stream_id;
      return *this;
    case Opcode::CudaStreamBarrier:
      return *this;
    default:
      std::ostringstream out;
      out << "Invalid instruction " << static_cast<int>(instr.op);
      throw std::runtime_error(out.str());
  }
}

Instruction::~Instruction() {
  switch (this->op) {
    case Opcode::Move:
    case Opcode::Ret:
    case Opcode::AllocTensorReg:
    case Opcode::If:
    case Opcode::LoadConst:
    case Opcode::GetField:
    case Opcode::Goto:
    case Opcode::LoadConsti:
    case Opcode::AllocStorage:
    case Opcode::Free:
    case Opcode::SetShape:
    case Opcode::Fatal:
    case Opcode::CudaSetStream:
    case Opcode::CudaAddEvent:
    case Opcode::CudaWaitEvent:
    case Opcode::CudaStreamBarrier:
      return;
    case Opcode::AllocTensor:
      delete[] this->alloc_tensor.shape;
      return;
    case Opcode::AllocTuple:
      delete[] this->alloc_tuple.fields;
      return;
    case Opcode::AllocClosure:
      delete[] this->alloc_closure.free_vars;
      return;
    case Opcode::InvokePacked:
      delete[] this->invoke_packed.args;
      return;
    case Opcode::InvokeClosure:
      delete[] this->invoke_closure.args;
      return;
    case Opcode::InvokeFunc:
      delete[] this->invoke_func.args;
      return;
    case Opcode::InvokeJit:
      delete[] this->invoke_jit.args;
      return;
    case Opcode::InferType:
      delete[] this->infer_type.args;
      return;
    default:
      std::ostringstream out;
      LOG(FATAL) << "Invalid instruction " << static_cast<int>(this->op);
  }
}

Instruction Instruction::Ret(RegName result) {
  Instruction instr;
  instr.op = Opcode::Ret;
  instr.result = result;
  return instr;
}

Instruction Instruction::Fatal() {
  Instruction instr;
  instr.op = Opcode::Fatal;
  return instr;
}

Instruction Instruction::InvokePacked(Index packed_index, Index arity, Index output_size,
                                      const std::vector<RegName>& args) {
  Instruction instr;
  instr.op = Opcode::InvokePacked;
  instr.invoke_packed.packed_index = packed_index;
  instr.invoke_packed.arity = arity;
  instr.invoke_packed.output_size = output_size;
  instr.invoke_packed.args = new RegName[arity];
  for (Index i = 0; i < arity; ++i) {
    instr.invoke_packed.args[i] = args[i];
  }
  return instr;
}

Instruction Instruction::AllocTensor(RegName storage, Index offset,
                                     const std::vector<int64_t>& shape, DLDataType dtype, Index dst,
                                     bool own) {
  Instruction instr;
  instr.op = Opcode::AllocTensor;
  instr.dst = dst;
  instr.alloc_tensor.storage = storage;
  instr.alloc_tensor.offset = offset;
  instr.alloc_tensor.ndim = shape.size();
  instr.alloc_tensor.shape = new int64_t[shape.size()];
  for (size_t i = 0; i < shape.size(); ++i) {
    instr.alloc_tensor.shape[i] = shape[i];
  }
  instr.alloc_tensor.dtype = dtype;
  instr.alloc_tensor.own = own;
  return instr;
}

Instruction Instruction::AllocTensorReg(RegName storage, Index offset, RegName shape_register,
                                        DLDataType dtype, Index dst, bool own) {
  Instruction instr;
  instr.op = Opcode::AllocTensorReg;
  instr.dst = dst;
  instr.alloc_tensor_reg.storage = storage;
  instr.alloc_tensor_reg.offset = offset;
  instr.alloc_tensor_reg.shape_register = shape_register;
  instr.alloc_tensor_reg.dtype = dtype;
  instr.alloc_tensor_reg.own = own;
  return instr;
}

Instruction Instruction::AllocStorage(RegName size, Index alignment, DLDataType dtype_hint,
                                      DevType device_type, Index device_id, Index dst,
                                      bool alloc_async) {
  Instruction instr;
  instr.op = Opcode::AllocStorage;
  instr.dst = dst;
  instr.alloc_storage.allocation_size = size;
  instr.alloc_storage.alignment = alignment;
  instr.alloc_storage.dtype_hint = dtype_hint;
  instr.alloc_storage.device_type = device_type;
  instr.alloc_storage.device_id = device_id;
  instr.alloc_storage.alloc_async = alloc_async;
  return instr;
}

Instruction Instruction::Free(RegName memory) {
  Instruction instr;
  instr.op = Opcode::Free;
  instr.free.memory = memory;
  return instr;
}

Instruction Instruction::AllocTuple(const std::vector<RegName>& fields, Index dst) {
  Instruction instr;
  instr.op = Opcode::AllocTuple;
  instr.dst = dst;
  instr.alloc_tuple.num_fields = fields.size();
  instr.alloc_tuple.fields = new RegName[fields.size()];
  for (Index i = 0; i < fields.size(); ++i) {
    instr.alloc_tuple.fields[i] = fields[i];
  }
  return instr;
}

Instruction Instruction::AllocClosure(Index func_index, const std::vector<RegName>& free_vars,
                                      Index dst) {
  Instruction instr;
  instr.op = Opcode::AllocClosure;
  instr.dst = dst;
  instr.alloc_closure.func_index = func_index;
  instr.alloc_closure.num_free_vars = free_vars.size();
  instr.alloc_closure.free_vars = new RegName[free_vars.size()];
  for (Index i = 0; i < free_vars.size(); ++i) {
    instr.alloc_closure.free_vars[i] = free_vars[i];
  }
  return instr;
}

Instruction Instruction::SetShape(RegName data, RegName shape, RegName dst) {
  Instruction instr;
  instr.op = Opcode::SetShape;
  instr.dst = dst;
  instr.set_shape.data = data;
  instr.set_shape.shape = shape;
  return instr;
}

Instruction Instruction::GetField(RegName object, Index field_index, RegName dst) {
  Instruction instr;
  instr.op = Opcode::GetField;
  instr.dst = dst;
  instr.get_field.object = object;
  instr.get_field.field_index = field_index;
  return instr;
}

Instruction Instruction::If(RegName test, RegName target, Index true_branch, Index false_branch) {
  Instruction instr;
  instr.op = Opcode::If;
  instr.if_op.test = test;
  instr.if_op.target = target;
  instr.if_op.true_offset = true_branch;
  instr.if_op.false_offset = false_branch;
  return instr;
}

Instruction Instruction::Goto(Index pc_offset) {
  Instruction instr;
  instr.op = Opcode::Goto;
  instr.pc_offset = pc_offset;
  return instr;
}

Instruction Instruction::InvokeFunc(Index func_index, const std::vector<RegName>& args,
                                    RegName dst) {
  Instruction instr;
  instr.op = Opcode::InvokeFunc;
  instr.dst = dst;
  instr.invoke_func.func_index = func_index;
  instr.invoke_func.num_args = args.size();
  instr.invoke_func.args = new RegName[instr.invoke_func.num_args];
  for (Index i = 0; i < instr.invoke_func.num_args; ++i) {
    instr.invoke_func.args[i] = args[i];
  }
  return instr;
}

Instruction Instruction::InvokeClosure(RegName closure, const std::vector<RegName>& args,
                                       RegName dst) {
  Instruction instr;
  instr.op = Opcode::InvokeClosure;
  instr.dst = dst;
  instr.invoke_closure.closure = closure;
  instr.invoke_closure.num_args = args.size();
  instr.invoke_closure.args = new RegName[args.size()];
  for (size_t i = 0; i < args.size(); ++i) {
    instr.invoke_closure.args[i] = args[i];
  }
  return instr;
}

Instruction Instruction::LoadConst(Index const_index, RegName dst) {
  Instruction instr;
  instr.op = Opcode::LoadConst;
  instr.dst = dst;
  instr.const_index = const_index;
  return instr;
}

Instruction Instruction::LoadConsti(Index val, RegName dst) {
  Instruction instr;
  instr.op = Opcode::LoadConsti;
  instr.dst = dst;
  instr.load_consti.val = val;
  return instr;
}

Instruction Instruction::Move(RegName src, RegName dst) {
  Instruction instr;
  instr.op = Opcode::Move;
  instr.dst = dst;
  instr.from = src;
  return instr;
}

Instruction Instruction::InvokeJit(RegName op_reg, Index arity, Index output_size,
                                   const std::vector<RegName>& args) {
  Instruction instr;
  instr.op = Opcode::InvokeJit;
  instr.invoke_jit.op_reg = op_reg;
  instr.invoke_jit.arity = arity;
  instr.invoke_jit.output_size = output_size;
  instr.invoke_jit.args = new RegName[arity];
  for (Index i = 0; i < arity; ++i) {
    instr.invoke_jit.args[i] = args[i];
  }
  return instr;
}

Instruction Instruction::InferType(RegName op_reg, const std::vector<RegName>& args, RegName dst) {
  Instruction instr;
  instr.op = Opcode::InferType;
  instr.dst = dst;
  instr.infer_type.op_reg = op_reg;
  instr.infer_type.num_args = args.size();
  instr.infer_type.args = new RegName[args.size()];
  for (Index i = 0; i < args.size(); ++i) {
    instr.infer_type.args[i] = args[i];
  }
  return instr;
}

Instruction Instruction::CudaSetStream(Index device_id, Index stream_id) {
  Instruction instr;
  instr.op = Opcode::CudaSetStream;
  instr.cuda_set_stream.device_id = device_id;
  instr.cuda_set_stream.stream_id = stream_id;
  return instr;
}

Instruction Instruction::CudaAddEvent(Index event_id, Index stream_id) {
  Instruction instr;
  instr.op = Opcode::CudaAddEvent;
  instr.cuda_event.event_id = event_id;
  instr.cuda_event.stream_id = stream_id;
  return instr;
}

Instruction Instruction::CudaWaitEvent(Index event_id, Index stream_id) {
  Instruction instr;
  instr.op = Opcode::CudaWaitEvent;
  instr.cuda_event.event_id = event_id;
  instr.cuda_event.stream_id = stream_id;
  return instr;
}

Instruction Instruction::CudaStreamBarrier() {
  Instruction instr;
  instr.op = Opcode::CudaStreamBarrier;
  return instr;
}

void DLDatatypePrint(std::ostream& os, const DLDataType& dtype) {
  switch (dtype.code) {
    case kDLInt:
      os << "int";
      break;
    case kDLUInt:
      os << "uint";
      break;
    case kDLFloat:
      os << "float";
      break;
  }

  os << int(dtype.bits);
  if (dtype.lanes != 1) {
    os << "x" << dtype.lanes;
  }
}

template <typename T>
std::string StrJoin(T* items, int offset, int cnt, std::string delim = ", ") {
  if (cnt == 0) {
    return "";
  }
  std::ostringstream oss;
  oss << items[offset];
  for (int i = 1; i < cnt; ++i) {
    oss << delim << items[offset + i];
  }
  return oss.str();
}

void InstructionPrint(std::ostream& os, const Instruction& instr) {
  switch (instr.op) {
    case Opcode::Move: {
      os << "move $" << instr.dst << " $" << instr.from;
      break;
    }
    case Opcode::Ret: {
      os << "ret $" << instr.result;
      break;
    }
    case Opcode::Fatal: {
      os << "fatal";
      break;
    }
    case Opcode::InvokePacked: {
      Index num_inputs = instr.invoke_packed.arity - instr.invoke_packed.output_size;
      os << "invoke_packed PackedFunc[" << instr.invoke_packed.packed_index << "] (in: $"
         << StrJoin<RegName>(instr.invoke_packed.args, 0, num_inputs, ", $") << ", out: $"
         << StrJoin<RegName>(instr.invoke_packed.args, num_inputs, instr.invoke_packed.output_size,
                             ", $")
         << ")";
      break;
    }
    case Opcode::AllocTensor: {
      os << "alloc_tensor $" << instr.dst << " $" << instr.alloc_tensor.storage << " $"
         << instr.alloc_tensor.offset << " ["
         << StrJoin<int64_t>(instr.alloc_tensor.shape, 0, instr.alloc_tensor.ndim) << "] ";
      DLDatatypePrint(os, instr.alloc_tensor.dtype);
      if (instr.alloc_tensor.own) {
        os << "(own)";
      }
      break;
    }
    case Opcode::AllocTensorReg: {
      os << "alloc_tensor_reg $" << instr.dst << " $" << instr.alloc_tensor_reg.storage << " $"
         << instr.alloc_tensor_reg.offset << " $" << instr.alloc_tensor_reg.shape_register << " ";
      DLDatatypePrint(os, instr.alloc_tensor_reg.dtype);
      if (instr.alloc_tensor_reg.own) {
        os << "(own)";
      }
      break;
    }
    case Opcode::AllocTuple: {
      os << "alloc_tuple $" << instr.dst << " [$"
         << StrJoin<RegName>(instr.alloc_tuple.fields, 0, instr.alloc_tuple.num_fields, ",$")
         << "]";
      break;
    }
    case Opcode::AllocClosure: {
      os << "alloc_closure $" << instr.dst << " VMFunc[" << instr.alloc_closure.func_index << "]($"
         << StrJoin<RegName>(instr.alloc_closure.free_vars, 0, instr.alloc_closure.num_free_vars,
                             ",$")
         << ")";
      break;
    }
    case Opcode::SetShape: {
      os << "set_shape $" << instr.dst << " $" << instr.set_shape.data << " $"
         << instr.set_shape.shape;
      break;
    }
    case Opcode::If: {
      os << "if "
         << "$" << instr.if_op.test << " $" << instr.if_op.target << " " << instr.if_op.true_offset
         << " " << instr.if_op.false_offset;
      break;
    }
    case Opcode::InvokeFunc: {
      os << "invoke_func $" << instr.dst << " VMFunc[" << instr.invoke_func.func_index << "]($"
         << StrJoin<RegName>(instr.invoke_func.args, 0, instr.invoke_func.num_args, ",$") << ")";
      break;
    }
    case Opcode::InvokeClosure: {
      os << "invoke_closure $" << instr.dst << " $" << instr.invoke_closure.closure << "($"
         << StrJoin<RegName>(instr.invoke_closure.args, 0, instr.invoke_closure.num_args, ",$")
         << ")";
      break;
    }
    case Opcode::LoadConst: {
      os << "load_const $" << instr.dst << " Const[" << instr.const_index << "]";
      break;
    }
    case Opcode::LoadConsti: {
      os << "load_consti $" << instr.dst << " " << instr.load_consti.val;
      break;
    }
    case Opcode::GetField: {
      os << "get_field $" << instr.dst << " $" << instr.get_field.object << "["
         << instr.get_field.field_index << "]";
      break;
    }
    case Opcode::Goto: {
      os << "goto " << instr.pc_offset;
      break;
    }
    case Opcode::AllocStorage: {
      os << "alloc_storage $" << instr.dst << " $" << instr.alloc_storage.allocation_size << " "
         << instr.alloc_storage.alignment << " "
         << tvm::runtime::DLDataType2String(instr.alloc_storage.dtype_hint);
      if (instr.alloc_storage.alloc_async) {
        os << "(async)";
      }
      break;
    }
    case Opcode::Free: {
      os << "free $" << instr.free.memory;
      break;
    }
    case Opcode::InvokeJit: {
      Index num_inputs = instr.invoke_jit.arity - instr.invoke_jit.output_size;
      os << "invoke_jit $" << instr.invoke_jit.op_reg << " (in: $"
         << StrJoin<RegName>(instr.invoke_jit.args, 0, num_inputs, ", $") << ", out: $"
         << StrJoin<RegName>(instr.invoke_jit.args, num_inputs, instr.invoke_jit.output_size, ", $")
         << ")";
      break;
    }
    case Opcode::InferType: {
      os << "infer_type $" << instr.dst << " $" << instr.infer_type.op_reg << "($"
         << StrJoin<RegName>(instr.infer_type.args, 0, instr.infer_type.num_args, ",$") << ")";
      break;
    }
    case Opcode::CudaSetStream: {
      os << "cuda_set_stream " << instr.cuda_set_stream.device_id << " "
         << instr.cuda_set_stream.stream_id;
      break;
    }
    case Opcode::CudaAddEvent: {
      os << "cuda_add_event " << instr.cuda_event.event_id << " " << instr.cuda_event.stream_id;
      break;
    }
    case Opcode::CudaWaitEvent: {
      os << "cuda_wait_event" << instr.cuda_event.event_id << " " << instr.cuda_event.stream_id;
      break;
    }
    case Opcode::CudaStreamBarrier: {
      os << "cuda_stream_barrier";
    }
    default:
      LOG(FATAL) << "should never hit this case" << static_cast<int>(instr.op);
      break;
  }
}

std::ostream& operator<<(std::ostream& os, const Instruction& instr) {
  InstructionPrint(os, instr);
  return os;
}

void VMFunctionPrint(std::ostream& os, const VMFunction& vm_func) {
  os << vm_func.name << ": " << std::endl;
  for (size_t i = 0; i < vm_func.instructions.size(); ++i) {
    os << i << ": " << vm_func.instructions[i] << ";" << std::endl;
  }
}

std::ostream& operator<<(std::ostream& os, const VMFunction& vm_func) {
  VMFunctionPrint(os, vm_func);
  return os;
}

}  // namespace vm
}  // namespace executor
}  // namespace raf
