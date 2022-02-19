/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/op/dialect/cuda/memory.cc
 * \brief Tensor fusion and defusion operators with asynchronous CUDA memory copy.
 */
#include <cuda_runtime.h>
#include <vector>
#include "mnm/op_utils.h"
#include "mnm/stream_pool.h"
#include "../../schema/memory.h"
#include "../../../common/shape_utils.h"
#include "../../../common/cuda_utils.h"

namespace mnm {
namespace op {
namespace cuda {

using namespace mnm::op::schema;
using mnm::common::shape_utils::BytesCompactTensor;
using mnm::stream_pool::StreamTagEnum;

class CudaFuseTensor : public mnm::op::OpEnv {
  void* stream;
  std::vector<int64_t> tuple_sizes;

  explicit CudaFuseTensor(const CallValues& cv) {
    auto op = ir::Op::Get("mnm.op.fuse_tensor");
    auto fschema_index = ir::Op::GetAttrMap<op::FMNMSchemaFieldIndex>("FMNMSchemaFieldIndex");
    this->arg_indices = {fschema_index[op]("data")};
    RequestStream(&stream, cv->device, StreamTagEnum::MemCudaToCuda1());

    auto args = cv->args.as<FuseTensorArgs>();
    auto& tv = args->data;
    tuple_sizes.clear();
    for (int i = 0; i < tv.size(); ++i) {
      DLTensor* x = tv[i];
      tuple_sizes.push_back(BytesCompactTensor(*x));
    }
  }

 public:
  ~CudaFuseTensor() {
    // Nothing
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("mnm.op.cuda.fuse_tensor"));
  }

  void Execute(const CallValues& cv) override {
    auto args = cv->args.as<FuseTensorArgs>();
    Execute({TupleValue::make(ir::Array<Value>(args->data.begin(), args->data.end()))}, cv->out);
  }

  void Execute(const std::vector<value::Value>& inputs, value::Value output) override {
    // Fuse Tensor
    DLTensor* out = output;
    auto tv = Downcast<value::TupleValue>(inputs[0]);
    size_t offset = 0;
    for (int i = 0; i < tv->fields.size(); ++i) {
      DLTensor* x = tv->fields[i];
      void* buffer_data_at_offset = reinterpret_cast<uint8_t*>(out->data) + offset;
      CUDA_CALL(cudaMemcpyAsync(buffer_data_at_offset, x->data, tuple_sizes[i],
                                cudaMemcpyDeviceToDevice, (cudaStream_t)stream));
      offset += tuple_sizes[i];
    }
  }

  static OpEnv* make(const CallValues& cv) {
    return new CudaFuseTensor(cv);
  }
};

MNM_REGISTER_DIALECT_OP(cuda, fuse_tensor, 10);
MNM_OP_ENV_MAKER("mnm.op.cuda.fuse_tensor", CudaFuseTensor::make);

class CudaDefuseTensor : public mnm::op::OpEnv {
  void* stream;
  std::vector<int64_t> tuple_sizes;

  explicit CudaDefuseTensor(const CallValues& cv) {
    auto op = ir::Op::Get("mnm.op.defuse_tensor");
    auto fschema_index = ir::Op::GetAttrMap<op::FMNMSchemaFieldIndex>("FMNMSchemaFieldIndex");
    this->arg_indices = {fschema_index[op]("data")};
    RequestStream(&stream, cv->device, StreamTagEnum::MemCudaToCuda2());

    tuple_sizes = cv->args.as<DefuseTensorArgs>()->sizes;
    DLTensor* x = cv->args.as<DefuseTensorArgs>()->data;
    int64_t nbytes = (x->dtype.bits + 7) / 8;
    for (auto& size : tuple_sizes) {
      size *= nbytes;
    }
  }

 public:
  ~CudaDefuseTensor() {
    // Nothing
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("mnm.op.cuda.defuse_tensor"));
  }

  void Execute(const CallValues& cv) {
    auto args = cv->args.as<DefuseTensorArgs>();
    Execute({args->data}, cv->out);
  }

  void Execute(const std::vector<value::Value>& inputs, value::Value output) {
    // Defuse Tensor
    DLTensor* in = inputs[0];
    int64_t nbytes = (in->dtype.bits + 7) / 8;
    auto& of = Downcast<value::TupleValue>(output)->fields;
    size_t offset = 0;
    for (int i = 0; i < tuple_sizes.size(); ++i) {
      DLTensor* x = of[i];
      void* buffer_data_at_offset = reinterpret_cast<uint8_t*>(in->data) + offset;
      CUDA_CALL(cudaMemcpyAsync(x->data, buffer_data_at_offset, tuple_sizes[i],
                                cudaMemcpyDeviceToDevice, (cudaStream_t)stream));
      offset += tuple_sizes[i];
    }
  }

  static OpEnv* make(const CallValues& cv) {
    return new CudaDefuseTensor(cv);
  }
};

MNM_REGISTER_DIALECT_OP(cuda, defuse_tensor, 10);
MNM_OP_ENV_MAKER("mnm.op.cuda.defuse_tensor", CudaDefuseTensor::make);

}  // namespace cuda
}  // namespace op
}  // namespace mnm
