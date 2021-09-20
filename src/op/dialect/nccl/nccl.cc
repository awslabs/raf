/*!
 * Copyright (c) 2021 by Contributors
 * \file src/op/dialect/cuda/nccl.cc
 * \brief Communication operators implmentated by NCCL
 */
#include <vector>
#include <chrono>
#include <thread>
#include "mnm/op_utils.h"
#include "mnm/dist_context.h"
#include "../../schema/communication.h"
#include "./communication_utils.h"

namespace mnm {
namespace op {
namespace communication {
namespace nccl {
using namespace distributed;
using namespace distributed::communicator;
using common::shape_utils::BytesCompactTensor;
using stream_pool::StreamTagEnum;

MNM_REGISTER_DIALECT("nccl").set_enable(DevType::kCUDA());

class NCCLAllReduce : public mnm::op::OpEnv {
  void* stream;
  void* communicator;
  void* fused_data;
  size_t total_size = 0;
  std::vector<size_t> tuple_sizes;
  DType dtype;
  ncclRedOp_t compute;

  explicit NCCLAllReduce(const CallValues& cv) {
    auto op = ir::Op::Get("mnm.op._allreduce");
    auto fschema_index = ir::Op::GetAttrMap<op::FMNMSchemaFieldIndex>("FMNMSchemaFieldIndex");
    this->arg_indices = {fschema_index[op]("x")};
    RequestStream(&stream, cv->device, StreamTagEnum::CudaCommunicate());
    RequestDistributed(&communicator);
    auto args = cv->args.as<mnm::op::schema::AllreduceArgs>();
    auto& tv = args->x;

    if (args->computation.compare("sum") == 0) {
      compute = ncclSum;
    } else if (args->computation.compare("prod") == 0) {
      compute = ncclProd;
    } else if (args->computation.compare("min") == 0) {
      compute = ncclMin;
    } else if (args->computation.compare("max") == 0) {
      compute = ncclMax;
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
    return TruncateName(GetUniqueName("mnm.op.nccl._allreduce"));
  }

  void Execute(const CallValues& cv) override {
    auto args = cv->args.as<mnm::op::schema::AllreduceArgs>();
    Execute({TupleValue::make(ir::Array<Value>(args->x.begin(), args->x.end()))}, cv->out);
  }

  void Execute(const std::vector<value::Value>& inputs, value::Value output) override {
    // We can use sleep to test communication scheduling locally.
    // using namespace std::this_thread;
    // using namespace std::chrono;
    // sleep_until(system_clock::now() + nanoseconds(200));
    void* nccl_comm = reinterpret_cast<Communicator*>(communicator)->GetCommHandle();

    // Fuse Tensor
    auto tv = Downcast<value::TupleValue>(inputs[0]);
    size_t dtype_size = 0;
    if (tv->fields.size() == 1) {
      DLTensor* x = tv->fields[0];
      DLTensor* out = output;
      dtype_size = sizeof(x->dtype);
      NCCL_CALL(ncclAllReduce(x->data, out->data, total_size / dtype_size, dtype, compute,
                              (ncclComm_t)nccl_comm, (cudaStream_t)stream));

    } else {
      size_t offset = 0;
      for (int i = 0; i < tv->fields.size(); ++i) {
        DLTensor* x = tv->fields[i];
        void* buffer_data_at_offset = reinterpret_cast<uint8_t*>(fused_data) + offset;
        cudaMemcpyAsync(buffer_data_at_offset, x->data, tuple_sizes[i], cudaMemcpyDeviceToDevice,
                        (cudaStream_t)stream);
        offset += tuple_sizes[i];
        dtype_size = sizeof(x->dtype);
      }

      // Allreduce
      NCCL_CALL(ncclAllReduce(fused_data, fused_data, total_size / dtype_size, ncclFloat, compute,
                              (ncclComm_t)nccl_comm, (cudaStream_t)stream));
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

MNM_REGISTER_DIALECT_OP(nccl, _allreduce, 10);
MNM_OP_ENV_MAKER("mnm.op.nccl._allreduce", NCCLAllReduce::make);

class NCCLAllGather : public mnm::op::OpEnv {
  void* stream;
  void* communicator;
  explicit NCCLAllGather(const CallValues& cv) {
    auto op = ir::Op::Get("mnm.op._allgather");
    auto fschema_index = ir::Op::GetAttrMap<op::FMNMSchemaFieldIndex>("FMNMSchemaFieldIndex");
    this->arg_indices = {fschema_index[op]("x")};
    RequestStream(&stream, cv->device, StreamTagEnum::CudaCommunicate());
    RequestDistributed(&communicator);
  }

 public:
  ~NCCLAllGather() {
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("mnm.op.nccl._allgather"));
  }

  void Execute(const CallValues& cv) {
    auto args = cv->args.as<mnm::op::schema::AllgatherArgs>();
    Execute({args->x}, cv->out);
  }

  void Execute(const std::vector<value::Value>& inputs, value::Value output) {
    void* nccl_comm = reinterpret_cast<Communicator*>(communicator)->GetCommHandle();
    DLTensor* x = inputs[0];
    DLTensor* out = output;
    int64_t size = 1;
    for (int i = 0; i < x->ndim; ++i) {
      size *= x->shape[i];
    }
    NCCL_CALL(ncclAllGather(x->data, out->data, size, DType(x->dtype), (ncclComm_t)nccl_comm,
                            (cudaStream_t)stream));
  }

  static OpEnv* make(const CallValues& cv) {
    return new NCCLAllGather(cv);
  }
};

MNM_REGISTER_DIALECT_OP(nccl, _allgather, 10);
MNM_OP_ENV_MAKER("mnm.op.nccl._allgather", NCCLAllGather::make);

class NCCLReduceScatter : public mnm::op::OpEnv {
  void* stream;
  void* communicator;
  void* in_buffer;
  size_t size_in_bytes;
  size_t size;

  explicit NCCLReduceScatter(const CallValues& cv) {
    auto op = ir::Op::Get("mnm.op._reduce_scatter");
    auto fschema_index = ir::Op::GetAttrMap<op::FMNMSchemaFieldIndex>("FMNMSchemaFieldIndex");
    this->arg_indices = {fschema_index[op]("x")};
    RequestStream(&stream, cv->device, StreamTagEnum::CudaCommunicate());
    RequestDistributed(&communicator);
    const DLTensor* out = cv->out;
    size_in_bytes = BytesCompactTensor(*out);
    size = size_in_bytes / (out->dtype.bits / 8);
    RequestWorkspace(&in_buffer, cv->device, size_in_bytes * DistContext::Global()->size);
  }

 public:
  ~NCCLReduceScatter() {
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("mnm.op.nccl._reduce_scatter"));
  }

  void Execute(const CallValues& cv) {
    auto args = cv->args.as<mnm::op::schema::ReduceScatterArgs>();
    Execute(std::vector<value::Value>(args->x.begin(), args->x.end()), cv->out);
  }

  void Execute(const std::vector<value::Value>& inputs, value::Value output) {
    void* nccl_comm = reinterpret_cast<Communicator*>(communicator)->GetCommHandle();
    size_t offset = 0;
    DLTensor* out = output;
    DType dtype;
    for (size_t i = 0; i < inputs.size(); ++i) {
      const DLTensor* x = inputs[i];
      void* buffer_data_at_offset = reinterpret_cast<uint8_t*>(in_buffer) + size_in_bytes * i;
      cudaMemcpyAsync(buffer_data_at_offset, x->data, size_in_bytes, cudaMemcpyDeviceToDevice,
                      (cudaStream_t)stream);
      dtype = x->dtype;
    }
    NCCL_CALL(ncclReduceScatter(in_buffer, out->data, size, dtype, ncclSum, (ncclComm_t)nccl_comm,
                                (cudaStream_t)stream));
  }

  static OpEnv* make(const CallValues& cv) {
    return new NCCLReduceScatter(cv);
  }
};

MNM_REGISTER_DIALECT_OP(nccl, _reduce_scatter, 10);
MNM_OP_ENV_MAKER("mnm.op.nccl._reduce_scatter", NCCLReduceScatter::make);

class NCCLBroadcast : public mnm::op::OpEnv {
  void* stream;
  void* communicator;
  void* fused_data;
  size_t total_size = 0;
  std::vector<size_t> tuple_sizes;
  DType dtype;
  int root;

  explicit NCCLBroadcast(const CallValues& cv) {
    auto op = ir::Op::Get("mnm.op._broadcast");
    auto fschema_index = ir::Op::GetAttrMap<op::FMNMSchemaFieldIndex>("FMNMSchemaFieldIndex");
    this->arg_indices = {fschema_index[op]("x")};
    auto args = cv->args.as<mnm::op::schema::BroadcastArgs>();
    RequestStream(&stream, cv->device, StreamTagEnum::CudaCommunicate());
    RequestDistributed(&communicator);
    auto& tv = args->x;
    root = args->root;
    for (int i = 0; i < tv.size(); ++i) {
      DLTensor* x = tv[i];
      size_t size = BytesCompactTensor(*x);
      tuple_sizes.push_back(size);
      total_size += size;
      dtype = x->dtype;
    }
    if (tv.size() == 1) return;
    RequestWorkspace(&fused_data, cv->device, total_size);
  }

 public:
  ~NCCLBroadcast() {
    // Nothing
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("mnm.op.nccl._broadcast"));
  }

  void Execute(const CallValues& cv) override {
    auto args = cv->args.as<mnm::op::schema::BroadcastArgs>();
    Execute({TupleValue::make(ir::Array<Value>(args->x.begin(), args->x.end()))}, cv->out);
  }

  void Execute(const std::vector<value::Value>& inputs, value::Value output) {
    void* nccl_comm = reinterpret_cast<Communicator*>(communicator)->GetCommHandle();
    auto tv = Downcast<value::TupleValue>(inputs[0]);
    size_t dtype_size = 0;
    if (tv->fields.size() == 1) {
      DLTensor* x = tv->fields[0];
      DLTensor* out = output;
      dtype_size = sizeof(x->dtype);
      NCCL_CALL(ncclBroadcast(x->data, out->data, total_size / dtype_size, dtype, root,
                              (ncclComm_t)nccl_comm, (cudaStream_t)stream));
      return;
    }

    size_t offset = 0;
    for (int i = 0; i < tv->fields.size(); ++i) {
      DLTensor* x = tv->fields[i];
      void* buffer_data_at_offset = reinterpret_cast<uint8_t*>(fused_data) + offset;
      cudaMemcpyAsync(buffer_data_at_offset, x->data, tuple_sizes[i], cudaMemcpyDeviceToDevice,
                      (cudaStream_t)stream);
      offset += tuple_sizes[i];
      CHECK(dtype_size == 0 || dtype_size == sizeof(x->dtype))
          << "Broadcast requires tensors to be the same type.";
      dtype_size = sizeof(x->dtype);
    }

    NCCL_CALL(ncclBroadcast(fused_data, fused_data, total_size / dtype_size, dtype, root,
                            (ncclComm_t)nccl_comm, (cudaStream_t)stream));

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

  static OpEnv* make(const CallValues& cv) {
    return new NCCLBroadcast(cv);
  }
};

MNM_REGISTER_DIALECT_OP(nccl, _broadcast, 10);
MNM_OP_ENV_MAKER("mnm.op.nccl._broadcast", NCCLBroadcast::make);

class NCCLSend : public mnm::op::OpEnv {
  void* stream;
  void* communicator;
  int peer;

  explicit NCCLSend(const CallValues& cv) {
    auto op = ir::Op::Get("mnm.op._send");
    auto fschema_index = ir::Op::GetAttrMap<op::FMNMSchemaFieldIndex>("FMNMSchemaFieldIndex");
    this->arg_indices = {fschema_index[op]("x")};
    RequestStream(&stream, cv->device, StreamTagEnum::CudaCommunicate());
    RequestDistributed(&communicator);
    const auto* args = cv->args.as<mnm::op::schema::SendArgs>();
    CHECK(args);
    peer = args->peer;
  }

 public:
  ~NCCLSend() {
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("mnm.op.nccl._send"));
  }

  void Execute(const CallValues& cv) {
    const auto* args = cv->args.as<mnm::op::schema::SendArgs>();
    CHECK(args);
    Execute({args->x}, cv->out);
  }

  void Execute(const std::vector<value::Value>& inputs, value::Value output) {
    void* nccl_comm = reinterpret_cast<Communicator*>(communicator)->GetCommHandle();
    const DLTensor* x = inputs[0];
    NCCL_CALL(ncclSend(x->data, BytesCompactTensor(*x) / (x->dtype.bits / 8), DType(x->dtype), peer,
                       (ncclComm_t)nccl_comm, (cudaStream_t)stream));
  }

  static OpEnv* make(const CallValues& cv) {
    return new NCCLSend(cv);
  }
};

MNM_REGISTER_DIALECT_OP(nccl, _send, 10);
MNM_OP_ENV_MAKER("mnm.op.nccl._send", NCCLSend::make);

class NCCLRecv : public mnm::op::OpEnv {
  void* stream;
  void* communicator;
  int peer;
  std::vector<int64_t> shape;
  DType dtype;

  explicit NCCLRecv(const CallValues& cv) {
    RequestStream(&stream, cv->device, StreamTagEnum::CudaCommunicate());
    RequestDistributed(&communicator);
    const auto* args = cv->args.as<mnm::op::schema::RecvArgs>();
    CHECK(args);
    peer = args->peer;
    shape = args->shape;
    dtype = ir::String2DLDataType(args->dtype);
  }

 public:
  ~NCCLRecv() {
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("mnm.op.nccl._recv"));
  }

  void Execute(const CallValues& cv) {
    Execute({}, cv->out);
  }

  void Execute(const std::vector<value::Value>& inputs, value::Value output) {
    void* nccl_comm = reinterpret_cast<Communicator*>(communicator)->GetCommHandle();
    DLTensor* out = output;
    NCCL_CALL(ncclRecv(out->data, BytesCompactTensor(*out) / (out->dtype.bits / 8),
                       DType(out->dtype), peer, (ncclComm_t)nccl_comm, (cudaStream_t)stream));
  }

  static OpEnv* make(const CallValues& cv) {
    return new NCCLRecv(cv);
  }
};

MNM_REGISTER_DIALECT_OP(nccl, _recv, 10);
MNM_OP_ENV_MAKER("mnm.op.nccl._recv", NCCLRecv::make);

class NCCLReduce : public mnm::op::OpEnv {
  void* stream;
  void* communicator;
  ncclRedOp_t compute;
  int root;
  DType dtype;
  size_t total_size = 0;
  std::vector<size_t> tuple_sizes;
  void* fused_data;

  explicit NCCLReduce(const CallValues& cv) {
    auto op = ir::Op::Get("mnm.op._reduce");
    auto fschema_index = ir::Op::GetAttrMap<op::FMNMSchemaFieldIndex>("FMNMSchemaFieldIndex");
    this->arg_indices = {fschema_index[op]("x")};
    RequestStream(&stream, cv->device, StreamTagEnum::CudaCommunicate());
    RequestDistributed(&communicator);
    auto args = cv->args.as<mnm::op::schema::CommReduceArgs>();
    root = args->root;
    if (args->computation.compare("sum") == 0) {
      compute = ncclSum;
    } else if (args->computation.compare("prod") == 0) {
      compute = ncclProd;
    } else if (args->computation.compare("min") == 0) {
      compute = ncclMin;
    } else if (args->computation.compare("max") == 0) {
      compute = ncclMax;
    } else {
      LOG(FATAL) << "Invalid computation " << args->computation;
    }

    auto& tv = args->x;
    for (int i = 0; i < tv.size(); ++i) {
      DLTensor* x = tv[i];
      size_t size = BytesCompactTensor(*x);
      tuple_sizes.push_back(size);
      total_size += size;
      dtype = x->dtype;
    }
    if (tv.size() >= 1) {
      RequestWorkspace(&fused_data, cv->device, total_size);
    }
  }

 public:
  ~NCCLReduce() {
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("mnm.op.nccl._reduce"));
  }

  void Execute(const CallValues& cv) override {
    auto args = cv->args.as<mnm::op::schema::CommReduceArgs>();
    Execute({TupleValue::make(ir::Array<Value>(args->x.begin(), args->x.end()))}, cv->out);
  }

  void Execute(const std::vector<value::Value>& inputs, value::Value output) override {
    void* nccl_comm = reinterpret_cast<Communicator*>(communicator)->GetCommHandle();
    auto input_x = Downcast<value::TupleValue>(inputs[0]);
    size_t dtype_size = 0;
    if (input_x->fields.size() == 1) {
      DLTensor* x = input_x->fields[0];
      DLTensor* out = output;
      dtype_size = sizeof(x->dtype);

      size_t dtype_size = sizeof(x->dtype);
      NCCL_CALL(ncclReduce(x->data, out->data, total_size / dtype_size, dtype, compute, root,
                           (ncclComm_t)nccl_comm, (cudaStream_t)stream));
    } else {
      size_t offset = 0;
      for (int i = 0; i < input_x->fields.size(); ++i) {
        DLTensor* x = input_x->fields[i];
        void* buffer_data_at_offset = reinterpret_cast<uint8_t*>(fused_data) + offset;
        cudaMemcpyAsync(buffer_data_at_offset, x->data, tuple_sizes[i], cudaMemcpyDeviceToDevice,
                        (cudaStream_t)stream);
        offset += tuple_sizes[i];
        dtype_size = sizeof(x->dtype);
      }

      NCCL_CALL(ncclReduce(fused_data, fused_data, total_size / dtype_size, dtype, compute, root,
                           (ncclComm_t)nccl_comm, (cudaStream_t)stream));
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
    return new NCCLReduce(cv);
  }
};

MNM_REGISTER_DIALECT_OP(nccl, _reduce, 10);
MNM_OP_ENV_MAKER("mnm.op.nccl._reduce", NCCLReduce::make);

}  // namespace nccl
}  // namespace communication
}  // namespace op
}  // namespace mnm
