/*!
 * Copyright (c) 2021 by Contributors
 * \file src/op/dispatch/cuda/stream.cc
 * \brief Communication operators for cuda stream controlling.
 */
#include <vector>
#include "mnm/op_utils.h"
#include "../../schema/stream.h"
#include "./communication_utils.h"

namespace mnm {
namespace op {
namespace communication {
namespace nccl {
class CudaStreamSync : public mnm::op::OpEnv {
  void* stream;
  explicit CudaStreamSync(const CallValues& cv) {
    auto args = cv->args.as<mnm::op::schema::StreamArgs>();
    auto& stream_tag_id = args->stream_tag;
    RequestStream(&stream, cv->device, stream_tag_id);
  }

 public:
  ~CudaStreamSync() {
    // Nothing
  }
  void Execute(const CallValues& cv) {
    cudaStreamSynchronize((cudaStream_t)stream);
  }
  void Execute(const std::vector<value::Value>& inputs, value::Value output) {
    cudaStreamSynchronize((cudaStream_t)stream);
  }
  static OpEnv* make(const CallValues& cv) {
    return new CudaStreamSync(cv);
  }
};
MNM_OP_DISPATCH("mnm.op.stream_sync", CudaStreamSync::make, DevType::kCUDA(), "nccl_communication");

}  // namespace nccl
}  // namespace communication
}  // namespace op
}  // namespace mnm
