/*!
 * Copyright (c) 2020 by Contributors
 * \file src/op/dispatch/communication/nccl.cc
 * \brief Communication operators implmentated by NCCL
 */
#include "../../op_utils.h"
#include "../../schema/init.h"
#include "../../schema/likes.h"
#include "../../schema/list_args.h"
#include "../../schema/loss.h"
#include "../../schema/nn.h"
#include "../../schema/optimizer.h"
#include "../../schema/ufunc.h"
#include "../../schema/communication.h"
#include "./communication_utils.h"

namespace mnm {
namespace op {
namespace communication {
namespace nccl {
using common::shape_utils::BytesCompactTensor;
using common::shape_utils::GetShape;
using common::shape_utils::PadDims;
using common::shape_utils::Shape2Strides;
using distributed::communicator::Communicator;
using distributed::communicator::CommunicatorManager;
using stream_pool::StreamTagEnum;

using dmlc::BeginPtr;
using value::TupleValueObj;

class NCCLAllReduce : public mnm::op::OpEnv {
  void* stream;
  void* fused_data;
  size_t total_size = 0;
  std::vector<size_t> tuple_sizes;
  void* communicator;
  explicit NCCLAllReduce(const CallValues& cv) {
    auto args = cv->args.as<mnm::op::schema::AllreduceArgs>();
    RequestStream(&stream, cv->ctx, StreamTagEnum::CudaCommunicate());
    RequestDistributed(&communicator);
    auto& tv = args->x;
    for (int i = 0; i < tv.size(); ++i) {
      DLTensor* x = tv[i];
      size_t size = BytesCompactTensor(*x);
      tuple_sizes.push_back(size);
      total_size += size;
    }
    if (tv.size() == 1) return;
    RequestWorkspace(&fused_data, cv->ctx, total_size);
  }

 public:
  ~NCCLAllReduce() {
    // Nothing
  }
  void Execute(const CallValues& cv) {
    void* nccl_comm = reinterpret_cast<Communicator*>(communicator)->GetCommHandle();
    auto args = cv->args.as<mnm::op::schema::AllreduceArgs>();
    // Fuse Tensor
    auto& tv = args->x;
    size_t dtype_size = 0;
    if (tv.size() == 1) {
      DLTensor* x = tv[0];
      DLTensor* out = cv->out;
      dtype_size = sizeof(x->dtype);
      NCCL_CALL(ncclAllReduce(x->data, out->data, total_size / dtype_size, ncclFloat, ncclSum,
                              (ncclComm_t)nccl_comm, (cudaStream_t)stream));
      return;
    }
    size_t offset = 0;
    for (int i = 0; i < tv.size(); ++i) {
      DLTensor* x = tv[i];
      void* buffer_data_at_offset = reinterpret_cast<uint8_t*>(fused_data) + offset;
      cudaMemcpyAsync(buffer_data_at_offset, x->data, tuple_sizes[i], cudaMemcpyDeviceToDevice,
                      (cudaStream_t)stream);
      offset += tuple_sizes[i];
      dtype_size = sizeof(x->dtype);
    }

    // Allreduce
    NCCL_CALL(ncclAllReduce(fused_data, fused_data, total_size / dtype_size, ncclFloat, ncclSum,
                            (ncclComm_t)nccl_comm, (cudaStream_t)stream));

    // UnFuse Tensor
    value::TupleValue out = tvm::runtime::Downcast<value::TupleValue>(cv->out);
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
    return new NCCLAllReduce(cv);
  }
};
MNM_OP_DISPATCH("mnm.op._allreduce", NCCLAllReduce::make, DevType::kCUDA(), "nccl_communication");

}  // namespace nccl
}  // namespace communication
}  // namespace op
}  // namespace mnm
