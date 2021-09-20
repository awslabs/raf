/*!
 * Copyright (c) 2021 by Contributors
 * \file src/op/dialect/cuda/stream.cc
 * \brief Communication operators for cuda stream controlling.
 */
#include <cuda_runtime.h>
#include <vector>
#include "mnm/op_utils.h"
#include "../../schema/stream.h"

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
  std::string name() const override {
    return TruncateName(GetUniqueName("mnm.op.cuda.stream_sync"));
  }

  void Execute(const CallValues& cv) override {
    cudaStreamSynchronize((cudaStream_t)stream);
  }

  void Execute(const std::vector<value::Value>& inputs, value::Value output) override {
    cudaStreamSynchronize((cudaStream_t)stream);
  }

  static OpEnv* make(const CallValues& cv) {
    return new CudaStreamSync(cv);
  }
};

MNM_REGISTER_DIALECT_OP(cuda, stream_sync, 10);
MNM_OP_ENV_MAKER("mnm.op.cuda.stream_sync", CudaStreamSync::make);

}  // namespace nccl
}  // namespace communication
}  // namespace op
}  // namespace mnm
