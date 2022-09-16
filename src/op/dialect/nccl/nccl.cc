/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/dialect/cuda/nccl.cc
 * \brief Communication operators implmentated by NCCL
 */
#include <vector>
#include <chrono>
#include <thread>
#include "raf/op_utils.h"
#include "raf/dist_config.h"
#include "raf/nccl_communicator.h"
#include "../../../src/common/cuda_utils.h"
#include "../../schema/communication.h"
#include "./communication_utils.h"

namespace raf {
namespace op {
namespace communication {
namespace nccl {
using namespace distributed;
using namespace distributed::communicator;
using common::shape_utils::BytesCompactTensor;
using stream_pool::StreamTagEnum;

RAF_REGISTER_DIALECT("nccl").set_enable(DevType::kCUDA());

class NCCLOpEnv : public raf::op::OpEnv {
 protected:
  void* stream;
  void* communicator;
  explicit NCCLOpEnv(const CallValues& cv) {
    CUDA_CALL(cudaSetDevice(cv->device.device_id()));
  }
};

class NCCLAllReduce : public NCCLOpEnv {
  void* fused_data;
  size_t total_size = 0;
  std::vector<size_t> tuple_sizes;
  DType dtype;
  ncclRedOp_t compute;

  explicit NCCLAllReduce(const CallValues& cv) : NCCLOpEnv(cv) {
    auto op = ir::Op::Get("raf.op._allreduce");
    auto fschema_index = ir::Op::GetAttrMap<op::FRAFSchemaFieldIndex>("FRAFSchemaFieldIndex");
    auto args = cv->args.as<raf::op::schema::AllreduceArgs>();
    this->arg_indices = {fschema_index[op]("x")};
    RequestStream(&stream, cv->device, StreamTagEnum::CudaCommunicate());
    RequestDistributed(&communicator, "nccl", args->rank_list);

    auto& tv = args->x;

    if (args->computation.compare("sum") == 0) {
      compute = ncclSum;
    } else if (args->computation.compare("prod") == 0) {
      compute = ncclProd;
    } else if (args->computation.compare("min") == 0) {
      compute = ncclMin;
    } else if (args->computation.compare("max") == 0) {
      compute = ncclMax;
    } else if (args->computation.compare("avg") == 0) {
#if NCCL_VERSION_CODE >= 21000
      compute = ncclAvg;
#else
      LOG(FATAL) << "AllReduce with avg is not supported in NCCL < 2.10";
#endif
    } else {
      LOG(FATAL) << "Invalid computation " << args->computation;
    }

    for (int i = 0; i < tv.size(); ++i) {
      DLTensor* x = tv[i];
      size_t size = BytesCompactTensor(*x);
      tuple_sizes.push_back(size);
      total_size += size;
      dtype = x->dtype;
    }
    if (tv.size() > 1) {
      RequestWorkspace(&fused_data, cv->device, total_size);
    }
  }

 public:
  ~NCCLAllReduce() {
    // Nothing
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.nccl._allreduce"));
  }

  void Execute(const CallValues& cv) override {
    auto args = cv->args.as<raf::op::schema::AllreduceArgs>();
    Execute({TupleValue::make(ir::Array<Value>(args->x.begin(), args->x.end()))}, cv->out);
  }

  void Execute(const std::vector<value::Value>& inputs, value::Value output) override {
    // We can use sleep to test communication scheduling locally.
    // using namespace std::this_thread;
    // using namespace std::chrono;
    // sleep_until(system_clock::now() + nanoseconds(200));
    auto comm_ref = GetRef<Communicator>(reinterpret_cast<CommunicatorObj*>(communicator));
    ncclComm_t nccl_comm = Downcast<NCCLCommunicator>(comm_ref)->nccl_comm;

    // Fuse Tensor
    auto tv = Downcast<value::TupleValue>(inputs[0]);
    size_t dtype_size = 0;
    if (tv->fields.size() == 1) {
      DLTensor* x = tv->fields[0];
      DLTensor* out = output;
      dtype_size = GetSizeInBytes(x->dtype);
      NCCL_CALL(ncclAllReduce(x->data, out->data, total_size / dtype_size, dtype, compute,
                              nccl_comm, (cudaStream_t)stream));

    } else {
      size_t offset = 0;
      for (int i = 0; i < tv->fields.size(); ++i) {
        DLTensor* x = tv->fields[i];
        void* buffer_data_at_offset = reinterpret_cast<uint8_t*>(fused_data) + offset;
        cudaMemcpyAsync(buffer_data_at_offset, x->data, tuple_sizes[i], cudaMemcpyDeviceToDevice,
                        (cudaStream_t)stream);
        offset += tuple_sizes[i];
        CHECK(dtype_size == 0 || dtype_size == GetSizeInBytes(x->dtype))
            << "AllReduce requires tensors to be the same type.";
        dtype_size = GetSizeInBytes(x->dtype);
        dtype = x->dtype;
      }

      // Allreduce
      NCCL_CALL(ncclAllReduce(fused_data, fused_data, total_size / dtype_size, dtype, compute,
                              nccl_comm, (cudaStream_t)stream));
      // UnFuse Tensor
      value::TupleValue out = tvm::runtime::Downcast<value::TupleValue>(output);
      auto& of = out->fields;
      for (int i = of.size() - 1; i >= 0; --i) {
        DLTensor* x = of[i];
        offset -= tuple_sizes[i];
        void* buffer_data_at_offset = reinterpret_cast<uint8_t*>(fused_data) + offset;
        cudaMemcpyAsync(x->data, buffer_data_at_offset, tuple_sizes[i], cudaMemcpyDeviceToDevice,
                        (cudaStream_t)stream);
      }
    }
  }

  static OpEnv* make(const CallValues& cv) {
    return new NCCLAllReduce(cv);
  }
};

RAF_REGISTER_DIALECT_OP(nccl, _allreduce, 10);
RAF_OP_ENV_MAKER("raf.op.nccl._allreduce", NCCLAllReduce::make);

class NCCLAllGather : public NCCLOpEnv {
  explicit NCCLAllGather(const CallValues& cv) : NCCLOpEnv(cv) {
    auto op = ir::Op::Get("raf.op._allgather");
    auto fschema_index = ir::Op::GetAttrMap<op::FRAFSchemaFieldIndex>("FRAFSchemaFieldIndex");
    auto args = cv->args.as<raf::op::schema::AllgatherArgs>();
    this->arg_indices = {fschema_index[op]("x")};
    RequestStream(&stream, cv->device, StreamTagEnum::CudaCommunicate());
    RequestDistributed(&communicator, "nccl", args->rank_list);
  }

 public:
  ~NCCLAllGather() {
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.nccl._allgather"));
  }

  void Execute(const CallValues& cv) {
    auto args = cv->args.as<raf::op::schema::AllgatherArgs>();
    Execute({args->x}, cv->out);
  }

  void Execute(const std::vector<value::Value>& inputs, value::Value output) {
    auto comm_ref = GetRef<Communicator>(reinterpret_cast<CommunicatorObj*>(communicator));
    ncclComm_t nccl_comm = Downcast<NCCLCommunicator>(comm_ref)->nccl_comm;
    DLTensor* x = inputs[0];
    DLTensor* out = output;
    int64_t size = 1;
    for (int i = 0; i < x->ndim; ++i) {
      size *= x->shape[i];
    }
    NCCL_CALL(
        ncclAllGather(x->data, out->data, size, DType(x->dtype), nccl_comm, (cudaStream_t)stream));
  }

  static OpEnv* make(const CallValues& cv) {
    return new NCCLAllGather(cv);
  }
};

RAF_REGISTER_DIALECT_OP(nccl, _allgather, 10);
RAF_OP_ENV_MAKER("raf.op.nccl._allgather", NCCLAllGather::make);

class NCCLGroupAllGather : public NCCLOpEnv {
  explicit NCCLGroupAllGather(const CallValues& cv) : NCCLOpEnv(cv) {
    auto op = ir::Op::Get("raf.op._group_allgather");
    auto fschema_index = ir::Op::GetAttrMap<op::FRAFSchemaFieldIndex>("FRAFSchemaFieldIndex");
    this->arg_indices = {fschema_index[op]("tensor_list")};
    RequestStream(&stream, cv->device, StreamTagEnum::CudaCommunicate());
    RequestDistributed(&communicator, "nccl", NullValue<Value>());
  }

 public:
  ~NCCLGroupAllGather() {
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.nccl._group_allgather"));
  }

  void Execute(const CallValues& cv) override {
    auto args = cv->args.as<raf::op::schema::GroupAllgatherArgs>();
    Execute(
        {TupleValue::make(ir::Array<Value>(args->tensor_list.begin(), args->tensor_list.end()))},
        cv->out);
  }

  void Execute(const std::vector<value::Value>& inputs, value::Value output) override {
    auto comm_ptr = reinterpret_cast<NCCLCommunicatorObj*>(communicator);
    ncclComm_t nccl_comm = comm_ptr->nccl_comm;
    value::TupleValue out = tvm::runtime::Downcast<value::TupleValue>(output);
    auto tv = Downcast<value::TupleValue>(inputs[0]);

    NCCL_CALL(ncclGroupStart());
    for (int ti = 0; ti < tv->fields.size(); ++ti) {
      DLTensor* it = tv->fields[ti];
      DLTensor* ot = out->fields[ti];
      int64_t size = 1;
      for (int i = 0; i < it->ndim; ++i) {
        size *= it->shape[i];
      }
      NCCL_CALL(ncclAllGather(it->data, ot->data, size, DType(it->dtype), nccl_comm,
                              (cudaStream_t)stream));
    }
    NCCL_CALL(ncclGroupEnd());
  }

  static OpEnv* make(const CallValues& cv) {
    return new NCCLGroupAllGather(cv);
  }
};

RAF_REGISTER_DIALECT_OP(nccl, _group_allgather, 10);
RAF_OP_ENV_MAKER("raf.op.nccl._group_allgather", NCCLGroupAllGather::make);

class NCCLReduceScatter : public NCCLOpEnv {
  size_t size;
  ncclRedOp_t compute;

  explicit NCCLReduceScatter(const CallValues& cv) : NCCLOpEnv(cv) {
    auto op = ir::Op::Get("raf.op._reduce_scatter");
    auto fschema_index = ir::Op::GetAttrMap<op::FRAFSchemaFieldIndex>("FRAFSchemaFieldIndex");
    this->arg_indices = {fschema_index[op]("x")};
    RequestStream(&stream, cv->device, StreamTagEnum::CudaCommunicate());
    auto args = cv->args.as<raf::op::schema::ReduceScatterArgs>();
    RequestDistributed(&communicator, "nccl", args->rank_list);
    if (args->computation.compare("sum") == 0) {
      compute = ncclSum;
    } else if (args->computation.compare("prod") == 0) {
      compute = ncclProd;
    } else if (args->computation.compare("min") == 0) {
      compute = ncclMin;
    } else if (args->computation.compare("max") == 0) {
      compute = ncclMax;
    } else if (args->computation.compare("avg") == 0) {
#if NCCL_VERSION_CODE >= 21000
      compute = ncclAvg;
#else
      LOG(FATAL) << "ReduceScatter with avg is not supported in NCCL < 2.10";
#endif
    } else {
      LOG(FATAL) << "Invalid computation " << args->computation;
    }

    const DLTensor* out = cv->out;
    size_t size_in_bytes = BytesCompactTensor(*out);
    size = size_in_bytes / (out->dtype.bits / 8);
  }

 public:
  ~NCCLReduceScatter() {
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.nccl._reduce_scatter"));
  }

  void Execute(const CallValues& cv) {
    auto args = cv->args.as<raf::op::schema::ReduceScatterArgs>();
    Execute({args->x}, cv->out);
  }

  void Execute(const std::vector<value::Value>& inputs, value::Value output) {
    auto comm_ref = GetRef<Communicator>(reinterpret_cast<CommunicatorObj*>(communicator));
    ncclComm_t nccl_comm = Downcast<NCCLCommunicator>(comm_ref)->nccl_comm;
    DLTensor* out = output;
    DType dtype;

    DLTensor* x = inputs[0];
    dtype = x->dtype;
    NCCL_CALL(ncclReduceScatter(x->data, out->data, size, dtype, compute, nccl_comm,
                                (cudaStream_t)stream));
  }

  static OpEnv* make(const CallValues& cv) {
    return new NCCLReduceScatter(cv);
  }
};

RAF_REGISTER_DIALECT_OP(nccl, _reduce_scatter, 10);
RAF_OP_ENV_MAKER("raf.op.nccl._reduce_scatter", NCCLReduceScatter::make);

class NCCLGroupReduceScatter : public NCCLOpEnv {
  std::vector<size_t> sizes;
  ncclRedOp_t compute;

  explicit NCCLGroupReduceScatter(const CallValues& cv) : NCCLOpEnv(cv) {
    auto op = ir::Op::Get("raf.op._group_reduce_scatter");
    auto fschema_index = ir::Op::GetAttrMap<op::FRAFSchemaFieldIndex>("FRAFSchemaFieldIndex");
    this->arg_indices = {fschema_index[op]("tensor_list")};
    RequestStream(&stream, cv->device, StreamTagEnum::CudaCommunicate());
    RequestDistributed(&communicator, "nccl", NullValue<Value>());
    auto args = cv->args.as<raf::op::schema::GroupReduceScatterArgs>();
    if (args->computation.compare("sum") == 0) {
      compute = ncclSum;
    } else if (args->computation.compare("prod") == 0) {
      compute = ncclProd;
    } else if (args->computation.compare("min") == 0) {
      compute = ncclMin;
    } else if (args->computation.compare("max") == 0) {
      compute = ncclMax;
    } else if (args->computation.compare("avg") == 0) {
#if NCCL_VERSION_CODE >= 21000
      compute = ncclAvg;
#else
      LOG(FATAL) << "ReduceScatter with avg is not supported in NCCL < 2.10";
#endif
    } else {
      LOG(FATAL) << "Invalid computation " << args->computation;
    }

    auto out = Downcast<value::TupleValue>(cv->out);
    for (auto tv : out->fields) {
      const DLTensor* ot = tv;
      size_t size = 1;
      for (int i = 0; i < ot->ndim; ++i) {
        size *= ot->shape[i];
      }
      sizes.push_back(size);
    }
  }

 public:
  ~NCCLGroupReduceScatter() {
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.nccl._group_reduce_scatter"));
  }

  void Execute(const CallValues& cv) override {
    auto args = cv->args.as<raf::op::schema::GroupReduceScatterArgs>();
    Execute(
        {TupleValue::make(ir::Array<Value>(args->tensor_list.begin(), args->tensor_list.end()))},
        cv->out);
  }

  void Execute(const std::vector<value::Value>& inputs, value::Value output) override {
    auto comm_ptr = reinterpret_cast<NCCLCommunicatorObj*>(communicator);
    ncclComm_t nccl_comm = comm_ptr->nccl_comm;
    auto tv = Downcast<value::TupleValue>(inputs[0]);
    auto out = Downcast<value::TupleValue>(output);
    size_t offset = 0;
    DType dtype;
    NCCL_CALL(ncclGroupStart());
    for (int ti = 0; ti < tv->fields.size(); ++ti) {
      DLTensor* x = tv->fields[ti];
      DLTensor* ot = out->fields[ti];
      dtype = x->dtype;
      NCCL_CALL(ncclReduceScatter(x->data, ot->data, sizes[ti], dtype, compute, nccl_comm,
                                  (cudaStream_t)stream));
    }
    NCCL_CALL(ncclGroupEnd());
  }

  static OpEnv* make(const CallValues& cv) {
    return new NCCLGroupReduceScatter(cv);
  }
};

RAF_REGISTER_DIALECT_OP(nccl, _group_reduce_scatter, 10);
RAF_OP_ENV_MAKER("raf.op.nccl._group_reduce_scatter", NCCLGroupReduceScatter::make);

class NCCLBroadcast : public NCCLOpEnv {
  int root;

  explicit NCCLBroadcast(const CallValues& cv) : NCCLOpEnv(cv) {
    auto op = ir::Op::Get("raf.op._broadcast");
    auto fschema_index = ir::Op::GetAttrMap<op::FRAFSchemaFieldIndex>("FRAFSchemaFieldIndex");
    this->arg_indices = {fschema_index[op]("x")};
    auto args = cv->args.as<raf::op::schema::BroadcastArgs>();
    RequestStream(&stream, cv->device, StreamTagEnum::CudaCommunicate());
    RequestDistributed(&communicator, "nccl", args->rank_list);
    root = args->root;
  }

 public:
  ~NCCLBroadcast() {
    // Nothing
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.nccl._broadcast"));
  }

  void Execute(const CallValues& cv) override {
    auto args = cv->args.as<raf::op::schema::BroadcastArgs>();
    Execute({args->x}, cv->out);
  }

  void Execute(const std::vector<value::Value>& inputs, value::Value output) {
    auto comm_ref = GetRef<Communicator>(reinterpret_cast<CommunicatorObj*>(communicator));
    ncclComm_t nccl_comm = Downcast<NCCLCommunicator>(comm_ref)->nccl_comm;
    const DLTensor* x = inputs[0];
    DLTensor* out = output;
    size_t total_size = BytesCompactTensor(*x);
    size_t dtype_size = GetSizeInBytes(x->dtype);
    NCCL_CALL(ncclBroadcast(x->data, out->data, total_size / dtype_size, DType(x->dtype), root,
                            nccl_comm, (cudaStream_t)stream));
  }

  static OpEnv* make(const CallValues& cv) {
    return new NCCLBroadcast(cv);
  }
};

RAF_REGISTER_DIALECT_OP(nccl, _broadcast, 10);
RAF_OP_ENV_MAKER("raf.op.nccl._broadcast", NCCLBroadcast::make);

class NCCLAllToAll : public raf::op::OpEnv {
  void* stream;
  void* communicator;
  DType dtype;

  explicit NCCLAllToAll(const CallValues& cv) {
    auto op = ir::Op::Get("raf.op._all_to_all");
    auto fschema_index = ir::Op::GetAttrMap<op::FRAFSchemaFieldIndex>("FRAFSchemaFieldIndex");
    this->arg_indices = {fschema_index[op]("x")};
    RequestStream(&stream, cv->device, StreamTagEnum::CudaCommunicate());
    auto args = cv->args.as<raf::op::schema::AllToAllArgs>();
    RequestDistributed(&communicator, "nccl", args->rank_list);
    DLTensor* x = args->x;
    dtype = x->dtype;
#if NCCL_VERSION_CODE < 20700
    LOG(FATAL) << "AllToAll is not supported in NCCL < 2.7.0";
#endif
  }

 public:
  ~NCCLAllToAll() {
    // Nothing
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.nccl._all_to_all"));
  }

  void Execute(const CallValues& cv) override {
    auto args = cv->args.as<raf::op::schema::AllToAllArgs>();
    Execute({args->x}, cv->out);
  }

  void Execute(const std::vector<Value>& inputs, value::Value output) {
    auto comm_ref = GetRef<Communicator>(reinterpret_cast<CommunicatorObj*>(communicator));
    ncclComm_t nccl_comm = Downcast<NCCLCommunicator>(comm_ref)->nccl_comm;
    const DLTensor* x = inputs[0];
    DLTensor* out = output;

    int nccl_num_ranks;
    NCCL_CALL(ncclCommCount(nccl_comm, &nccl_num_ranks));
    CHECK_EQ(comm_ref->size, nccl_num_ranks)
        << "NCCL communicator world size does not match with RAF Communicator.";

    NCCL_CALL(ncclGroupStart());

    int64_t size = 1;
    for (int i = 0; i < x->ndim; ++i) {
      size *= x->shape[i];
    }
    CHECK(size % nccl_num_ranks == 0) << "Cannot evenly distribute input tensor to all ranks.";
    int64_t dtype_size_in_bytes = GetSizeInBytes(x->dtype);

    size_t per_rank_bytes = size * dtype_size_in_bytes / nccl_num_ranks;
    size_t size_per_rank = per_rank_bytes / dtype_size_in_bytes;
    char* send_buffer = (char*)x->data;
    char* recv_buffer = (char*)out->data;
    if (size != 0) {
      for (size_t i = 0; i < nccl_num_ranks; i++) {
        NCCL_CALL(ncclSend(send_buffer + i * per_rank_bytes, size_per_rank, dtype, i, nccl_comm,
                           (cudaStream_t)stream));
        NCCL_CALL(ncclRecv(recv_buffer + i * per_rank_bytes, size_per_rank, dtype, i, nccl_comm,
                           (cudaStream_t)stream));
      }
    }
    NCCL_CALL(ncclGroupEnd());
  }

  static OpEnv* make(const CallValues& cv) {
    return new NCCLAllToAll(cv);
  }
};

RAF_REGISTER_DIALECT_OP(nccl, _all_to_all, 10);
RAF_OP_ENV_MAKER("raf.op.nccl._all_to_all", NCCLAllToAll::make);

class NCCLSend : public NCCLOpEnv {
  int peer;

  explicit NCCLSend(const CallValues& cv) : NCCLOpEnv(cv) {
    auto op = ir::Op::Get("raf.op._send");
    auto fschema_index = ir::Op::GetAttrMap<op::FRAFSchemaFieldIndex>("FRAFSchemaFieldIndex");
    this->arg_indices = {fschema_index[op]("x")};
    RequestStream(&stream, cv->device, StreamTagEnum::CudaCommunicate());
    RequestDistributed(&communicator, "nccl", NullValue<Value>());
    const auto* args = cv->args.as<raf::op::schema::SendArgs>();
    CHECK(args);
    peer = args->peer;
  }

 public:
  ~NCCLSend() {
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.nccl._send"));
  }

  void Execute(const CallValues& cv) {
    const auto* args = cv->args.as<raf::op::schema::SendArgs>();
    CHECK(args);
    Execute({args->x}, cv->out);
  }

  void Execute(const std::vector<value::Value>& inputs, value::Value output) {
    auto comm_ref = GetRef<Communicator>(reinterpret_cast<CommunicatorObj*>(communicator));
    ncclComm_t nccl_comm = Downcast<NCCLCommunicator>(comm_ref)->nccl_comm;
    const DLTensor* x = inputs[0];
    NCCL_CALL(ncclSend(x->data, BytesCompactTensor(*x) / (x->dtype.bits / 8), DType(x->dtype), peer,
                       nccl_comm, (cudaStream_t)stream));
  }

  static OpEnv* make(const CallValues& cv) {
    return new NCCLSend(cv);
  }
};

RAF_REGISTER_DIALECT_OP(nccl, _send, 10);
RAF_OP_ENV_MAKER("raf.op.nccl._send", NCCLSend::make);

class NCCLRecv : public NCCLOpEnv {
  void* stream;
  void* communicator;
  int peer;
  std::vector<int64_t> shape;
  DType dtype;

  explicit NCCLRecv(const CallValues& cv) : NCCLOpEnv(cv) {
    RequestStream(&stream, cv->device, StreamTagEnum::CudaCommunicate());
    RequestDistributed(&communicator, "nccl", NullValue<Value>());
    const auto* args = cv->args.as<raf::op::schema::RecvArgs>();
    CHECK(args);
    peer = args->peer;
    shape = args->shape;
    dtype = ir::String2DLDataType(args->dtype);
  }

 public:
  ~NCCLRecv() {
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.nccl._recv"));
  }

  void Execute(const CallValues& cv) {
    Execute({}, cv->out);
  }

  void Execute(const std::vector<value::Value>& inputs, value::Value output) {
    auto comm_ref = GetRef<Communicator>(reinterpret_cast<CommunicatorObj*>(communicator));
    ncclComm_t nccl_comm = Downcast<NCCLCommunicator>(comm_ref)->nccl_comm;
    DLTensor* out = output;
    NCCL_CALL(ncclRecv(out->data, BytesCompactTensor(*out) / (out->dtype.bits / 8),
                       DType(out->dtype), peer, nccl_comm, (cudaStream_t)stream));
  }

  static OpEnv* make(const CallValues& cv) {
    return new NCCLRecv(cv);
  }
};

RAF_REGISTER_DIALECT_OP(nccl, _recv, 10);
RAF_OP_ENV_MAKER("raf.op.nccl._recv", NCCLRecv::make);

class NCCLReduce : public NCCLOpEnv {
  void* stream;
  void* communicator;
  ncclRedOp_t compute;
  int root;

  explicit NCCLReduce(const CallValues& cv) : NCCLOpEnv(cv) {
    auto op = ir::Op::Get("raf.op._reduce");
    auto fschema_index = ir::Op::GetAttrMap<op::FRAFSchemaFieldIndex>("FRAFSchemaFieldIndex");
    this->arg_indices = {fschema_index[op]("x")};
    RequestStream(&stream, cv->device, StreamTagEnum::CudaCommunicate());
    auto args = cv->args.as<raf::op::schema::CommReduceArgs>();
    RequestDistributed(&communicator, "nccl", args->rank_list);
    root = args->root;
    if (args->computation.compare("sum") == 0) {
      compute = ncclSum;
    } else if (args->computation.compare("prod") == 0) {
      compute = ncclProd;
    } else if (args->computation.compare("min") == 0) {
      compute = ncclMin;
    } else if (args->computation.compare("max") == 0) {
      compute = ncclMax;
    } else if (args->computation.compare("avg") == 0) {
#if NCCL_VERSION_CODE >= 21000
      compute = ncclAvg;
#else
      LOG(FATAL) << "Reduce with avg is not supported in NCCL < 2.10";
#endif
    } else {
      LOG(FATAL) << "Invalid computation " << args->computation;
    }
  }

 public:
  ~NCCLReduce() {
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.nccl._reduce"));
  }

  void Execute(const CallValues& cv) override {
    auto args = cv->args.as<raf::op::schema::CommReduceArgs>();
    Execute({args->x}, cv->out);
  }

  void Execute(const std::vector<value::Value>& inputs, value::Value output) override {
    auto comm_ref = GetRef<Communicator>(reinterpret_cast<CommunicatorObj*>(communicator));
    ncclComm_t nccl_comm = Downcast<NCCLCommunicator>(comm_ref)->nccl_comm;
    const DLTensor* x = inputs[0];
    DLTensor* out = output;
    size_t total_size = BytesCompactTensor(*x);
    size_t dtype_size = GetSizeInBytes(x->dtype);
    NCCL_CALL(ncclReduce(x->data, out->data, total_size / dtype_size, DType(x->dtype), compute,
                         root, nccl_comm, (cudaStream_t)stream));
  }

  static OpEnv* make(const CallValues& cv) {
    return new NCCLReduce(cv);
  }
};

RAF_REGISTER_DIALECT_OP(nccl, _reduce, 10);
RAF_OP_ENV_MAKER("raf.op.nccl._reduce", NCCLReduce::make);

class NCCLGather : public NCCLOpEnv {
  int root;
  DType dtype;
  explicit NCCLGather(const CallValues& cv) : NCCLOpEnv(cv) {
    auto op = ir::Op::Get("raf.op._gather");
    auto fschema_index = ir::Op::GetAttrMap<op::FRAFSchemaFieldIndex>("FRAFSchemaFieldIndex");
    auto args = cv->args.as<raf::op::schema::GatherScatterArgs>();
    this->arg_indices = {fschema_index[op]("x")};
    RequestStream(&stream, cv->device, StreamTagEnum::CudaCommunicate());
    RequestDistributed(&communicator, "nccl", NullValue<Value>());
    root = args->root;
    DLTensor* x = args->x;
    dtype = x->dtype;
#if NCCL_VERSION_CODE < 20700
    LOG(FATAL) << "Gather is not supported in NCCL < 2.7.0";
#endif
  }

 public:
  ~NCCLGather() {
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.nccl._gather"));
  }

  void Execute(const CallValues& cv) {
    auto args = cv->args.as<raf::op::schema::GatherScatterArgs>();
    Execute({args->x}, cv->out);
  }

  void Execute(const std::vector<value::Value>& inputs, value::Value output) {
    auto comm_ref = GetRef<Communicator>(reinterpret_cast<CommunicatorObj*>(communicator));
    ncclComm_t nccl_comm = Downcast<NCCLCommunicator>(comm_ref)->nccl_comm;
    const DLTensor* x = inputs[0];
    DLTensor* out = output;
    int nccl_num_ranks;
    int nccl_user_rank;
    NCCL_CALL(ncclCommCount(nccl_comm, &nccl_num_ranks));
    NCCL_CALL(ncclCommUserRank(nccl_comm, &nccl_user_rank));
    NCCL_CALL(ncclGroupStart());
    int64_t size = 1;
    for (int i = 0; i < x->ndim; ++i) {
      size *= x->shape[i];
    }
    int64_t dtype_size_in_bytes = GetSizeInBytes(dtype);
    size_t per_rank_bytes = size * dtype_size_in_bytes;
    char* send_buffer = (char*)x->data;
    char* recv_buffer = (char*)out->data;
    if (size != 0) {
      if (nccl_user_rank == root) {
        for (size_t i = 0; i < nccl_num_ranks; i++) {
          NCCL_CALL(ncclRecv(recv_buffer + i * per_rank_bytes, size, dtype, i, nccl_comm,
                             (cudaStream_t)stream));
        }
      }
      NCCL_CALL(ncclSend(send_buffer, size, dtype, root, nccl_comm, (cudaStream_t)stream));
    }
    NCCL_CALL(ncclGroupEnd());
  }

  static OpEnv* make(const CallValues& cv) {
    return new NCCLGather(cv);
  }
};

RAF_REGISTER_DIALECT_OP(nccl, _gather, 10);
RAF_OP_ENV_MAKER("raf.op.nccl._gather", NCCLGather::make);

class NCCLScatter : public NCCLOpEnv {
  int root;
  DType dtype;
  explicit NCCLScatter(const CallValues& cv) : NCCLOpEnv(cv) {
    auto op = ir::Op::Get("raf.op._scatter");
    auto fschema_index = ir::Op::GetAttrMap<op::FRAFSchemaFieldIndex>("FRAFSchemaFieldIndex");
    auto args = cv->args.as<raf::op::schema::GatherScatterArgs>();
    this->arg_indices = {fschema_index[op]("x")};
    RequestStream(&stream, cv->device, StreamTagEnum::CudaCommunicate());
    RequestDistributed(&communicator, "nccl", NullValue<Value>());
    root = args->root;
    DLTensor* x = args->x;
    dtype = x->dtype;
#if NCCL_VERSION_CODE < 20700
    LOG(FATAL) << "Scatter is not supported in NCCL < 2.7.0";
#endif
  }

 public:
  ~NCCLScatter() {
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.nccl._scatter"));
  }

  void Execute(const CallValues& cv) {
    auto args = cv->args.as<raf::op::schema::GatherScatterArgs>();
    Execute({args->x}, cv->out);
  }

  void Execute(const std::vector<value::Value>& inputs, value::Value output) {
    auto comm_ref = GetRef<Communicator>(reinterpret_cast<CommunicatorObj*>(communicator));
    ncclComm_t nccl_comm = Downcast<NCCLCommunicator>(comm_ref)->nccl_comm;
    const DLTensor* x = inputs[0];
    DLTensor* out = output;

    int nccl_num_ranks;
    int nccl_user_rank;
    NCCL_CALL(ncclCommCount(nccl_comm, &nccl_num_ranks));
    NCCL_CALL(ncclCommUserRank(nccl_comm, &nccl_user_rank));

    int64_t size = 1;
    for (int i = 0; i < x->ndim; ++i) {
      size *= x->shape[i];
    }
    int64_t dtype_size_in_bytes = GetSizeInBytes(dtype);

    size_t per_rank_bytes = size * dtype_size_in_bytes / nccl_num_ranks;
    size_t size_per_rank = per_rank_bytes / dtype_size_in_bytes;
    char* send_buffer = (char*)x->data;
    char* recv_buffer = (char*)out->data;
    NCCL_CALL(ncclGroupStart());
    if (nccl_user_rank == root) {
      for (size_t i = 0; i < nccl_num_ranks; i++) {
        NCCL_CALL(ncclSend(send_buffer + i * per_rank_bytes, size_per_rank, dtype, i, nccl_comm,
                           (cudaStream_t)stream));
      }
    }
    NCCL_CALL(ncclRecv(recv_buffer, size_per_rank, dtype, root, nccl_comm, (cudaStream_t)stream));
    NCCL_CALL(ncclGroupEnd());
  }

  static OpEnv* make(const CallValues& cv) {
    return new NCCLScatter(cv);
  }
};

RAF_REGISTER_DIALECT_OP(nccl, _scatter, 10);
RAF_OP_ENV_MAKER("raf.op.nccl._scatter", NCCLScatter::make);
}  // namespace nccl
}  // namespace communication
}  // namespace op
}  // namespace raf
