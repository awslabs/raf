/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/dialect/cudnn/dropout.cc
 * \brief cuDNN dropout operators.
 */
#include <tvm/support/random_engine.h>
#include "raf/ir_ext.h"
#include "raf/registry.h"
#include "raf/op_utils.h"
#include "raf/device.h"
#include "raf/device_api.h"
#include "../../../common/cuda_utils.h"
#include "../../schema/nn.h"
#include "./cudnn_utils.h"

namespace raf {
namespace op {
namespace cudnn {

using namespace raf::ir;
using namespace raf::memory_pool;
using namespace raf::value;

int64_t GetDropoutStateSizeInBytes() {
  size_t stateSizeInBytes;
  CUDNN_CALL(cudnnDropoutGetStatesSize(CUDNNThreadEntry::ThreadLocal()->handle, &stateSizeInBytes));
  return stateSizeInBytes;
}

class DropoutStatePool {
 public:
  explicit DropoutStatePool(const Device& dev) : device(dev) {
  }

  ~DropoutStatePool() {
    for (auto pair : state_pool) {
      if (pair.second != nullptr) {
        pair.second.reset();
      }
    }
  }

  static std::shared_ptr<DropoutStatePool> Get(const Device& dev) {
    static registry::PerDeviceStore<DropoutStatePool, false>* pool =
        new registry::PerDeviceStore<DropoutStatePool, false>();
    std::shared_ptr<DropoutStatePool>& ret = pool->Get(dev);
    if (ret == nullptr) {
      std::lock_guard<std::mutex> lock(pool->mutex_);
      if (ret == nullptr) {
        ret = std::make_shared<DropoutStatePool>(dev);
      }
    }
    return ret;
  }

  /*!
   * \brief Get an existing dropout state buffer of the given ratio.
   * If the state buffer for the ratio is not created, then this function initializes a new one.
   */
  std::pair<std::shared_ptr<Memory>, bool> GetState(float ratio) {
    bool init = false;
    std::lock_guard<std::mutex> lock(mutex);
    if (state_pool.count(ratio) == 0) {
      size_t stateSizeInBytes = GetDropoutStateSizeInBytes();
      state_pool[ratio] = memory_pool::Memory::Alloc(device, stateSizeInBytes);
      init = true;
    }
    return {state_pool[ratio], init};
  }

 public:
  Device device;
  std::unordered_map<float, std::shared_ptr<Memory>> state_pool;
  std::mutex mutex;
};

RAF_REGISTER_GLOBAL("raf.backend.cudnn.GetDropoutStateSizeInBytes")
    .set_body_typed(GetDropoutStateSizeInBytes);
RAF_REGISTER_GLOBAL("raf.backend.cudnn.GetDropoutState")
    .set_body_typed([](const Device& dev, double dropout) {
      auto state_n_init = DropoutStatePool::Get(dev)->GetState(dropout);
      auto buf = state_n_init.first;
      auto stateSizeInBytes = GetDropoutStateSizeInBytes();
      if (state_n_init.second) {
        // If the state buffer is newly created, then initialize it.
        cudnnDropoutDescriptor_t dropoutDesc;
        auto seed = tvm::support::LinearCongruentialEngine::DeviceRandom();
        CUDNN_CALL(cudnnCreateDropoutDescriptor(&dropoutDesc));
        CUDNN_CALL(cudnnSetDropoutDescriptor(dropoutDesc, CUDNNThreadEntry::ThreadLocal()->handle,
                                             dropout, buf->data, stateSizeInBytes, seed));
        CUDNN_CALL(cudnnDestroyDropoutDescriptor(dropoutDesc));
      }
      TensorValue state = TensorValue::Assemble(dev, DType(DTypeCode::kInt(), 8),
                                                {stateSizeInBytes}, {}, buf->data, buf);
      return state;
    });
RAF_REGISTER_GLOBAL("raf.backend.cudnn.GetDropoutReserveSpaceSizeInBytes")
    .set_body_typed([](TensorType x) {
      size_t reserveSpaceSizeInBytes;
      cudnnTensorDescriptor_t xdesc = NormalizeTensorType(x);
      CUDNN_CALL(cudnnDropoutGetReserveSpaceSize(xdesc, &reserveSpaceSizeInBytes));
      CUDNN_CALL(cudnnDestroyTensorDescriptor(xdesc));
      return (int64_t)reserveSpaceSizeInBytes;
    });

static auto fschema_index = ir::Op::GetAttrMap<op::FRAFSchemaFieldIndex>("FRAFSchemaFieldIndex");

class DropoutImplementedByCUDNNDropoutForward : public raf::op::OpEnv {
  cudnnDropoutDescriptor_t dropoutDesc;
  cudnnTensorDescriptor_t xdesc;
  cudnnTensorDescriptor_t ydesc;
  float dropout;
  size_t stateSizeInBytes;
  size_t reserveSpaceSizeInBytes;

  explicit DropoutImplementedByCUDNNDropoutForward(const CallValues& cv) {
    auto op = Op::Get("raf.op._contrib_dropout");
    auto args = cv->args.as<raf::op::schema::DropoutArgs>();
    dropout = args->p;
    TupleValue tv = Downcast<TupleValue>(cv->out);
    DLTensor* x = args->x;
    DLTensor* out = tv->fields[0];
    void* state_data = nullptr;

    // Note that we do not put "in_states" in arg_indices because we do not expect
    // in_states to be used in the VM.
    this->arg_indices = {fschema_index[op]("x")};

    bool is_first_dropout = false;
    if (args->in_states.get() == nullptr) {
      // If no state is provided, use the internal one.
      auto state_n_init = DropoutStatePool::Get(cv->device)->GetState(dropout);
      auto buf = state_n_init.first;
      is_first_dropout = state_n_init.second;
      state_data = buf->data;
    } else {
      DLTensor* state_tensor = args->in_states.value();
      state_data = state_tensor->data;
    }
    xdesc = NormalizeTensorType(SquashTensorShape(x, {}));
    ydesc = NormalizeTensorType(SquashTensorShape(out, {}));
    CUDNN_CALL(cudnnCreateDropoutDescriptor(&dropoutDesc));
    stateSizeInBytes = GetDropoutStateSizeInBytes();
    CUDNN_CALL(cudnnDropoutGetReserveSpaceSize(xdesc, &reserveSpaceSizeInBytes));

    if (is_first_dropout) {
      // Initial dropout desc with a certain ratio. This desc will be shared by all dropout ops
      // with the same ratio.
      auto seed = tvm::support::LinearCongruentialEngine::DeviceRandom();
      CUDNN_CALL(cudnnSetDropoutDescriptor(dropoutDesc, CUDNNThreadEntry::ThreadLocal()->handle,
                                           dropout, state_data, stateSizeInBytes, seed));
    } else {
      // The dropout desc has been globally initialized so we just restore it. Note that
      // in this case random seend has no effect so we simply put 0.
      CUDNN_CALL(cudnnRestoreDropoutDescriptor(dropoutDesc, CUDNNThreadEntry::ThreadLocal()->handle,
                                               dropout, state_data, stateSizeInBytes, 0));
    }
  }

 public:
  ~DropoutImplementedByCUDNNDropoutForward() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xdesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(ydesc));
    CUDNN_CALL(cudnnDestroyDropoutDescriptor(dropoutDesc));
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.cudnn._contrib_dropout"));
  }

  void Execute(const CallValues& cv) {
    auto args = cv->args.as<raf::op::schema::DropoutArgs>();
    CHECK(args != nullptr);
    TupleValue tv = Downcast<TupleValue>(cv->out);
    DLTensor* x = args->x;
    DLTensor* out = tv->fields[0];
    DLTensor* reserve_space = tv->fields[2];

    CUDNN_CALL(cudnnDropoutForward(CUDNNThreadEntry::ThreadLocal()->handle, dropoutDesc, xdesc,
                                   x->data, ydesc, out->data, reserve_space->data,
                                   reserveSpaceSizeInBytes));
  }

  void Execute(const std::vector<Value>& inputs, Value output) {
    CHECK_GE(inputs.size(), 1);
    TupleValue tv = Downcast<TupleValue>(output);
    DLTensor* x = inputs[0];
    DLTensor* out = tv->fields[0];
    DLTensor* reserve_space = tv->fields[2];

    CUDNN_CALL(cudnnDropoutForward(CUDNNThreadEntry::ThreadLocal()->handle, dropoutDesc, xdesc,
                                   x->data, ydesc, out->data, reserve_space->data,
                                   reserveSpaceSizeInBytes));
  }

  static OpEnv* make(const CallValues& cv) {
    return new DropoutImplementedByCUDNNDropoutForward(cv);
  }
};

RAF_REGISTER_DIALECT_OP(cudnn, _contrib_dropout, 15);
RAF_OP_ENV_MAKER("raf.op.cudnn._contrib_dropout", DropoutImplementedByCUDNNDropoutForward::make);

class DropoutImplementedByCUDNNDropoutBackward : public raf::op::OpEnv {
  cudnnDropoutDescriptor_t dropoutDesc;
  cudnnTensorDescriptor_t dxdesc;
  cudnnTensorDescriptor_t dydesc;
  float dropout;
  size_t stateSizeInBytes;
  size_t reserveSpaceSizeInBytes;

  explicit DropoutImplementedByCUDNNDropoutBackward(const CallValues& cv) {
    this->arg_indices = {/*dy=*/0, /*reserve_space=*/1};
    auto args = cv->args.as<raf::op::schema::DropoutDxArgs>();
    dropout = args->p;
    DLTensor* dx = cv->out;
    DLTensor* dy = args->dy;
    DLTensor* reserve_space = args->reserve_space;
    dxdesc = NormalizeTensorType(SquashTensorShape(dx, {}));
    dydesc = NormalizeTensorType(SquashTensorShape(dy, {}));
    reserveSpaceSizeInBytes = ComputeStorageInBytes(SquashTensorShape(reserve_space, {}));
    CUDNN_CALL(cudnnCreateDropoutDescriptor(&dropoutDesc));
    CUDNN_CALL(
        cudnnDropoutGetStatesSize(CUDNNThreadEntry::ThreadLocal()->handle, &stateSizeInBytes));

    void* state_data = DropoutStatePool::Get(cv->device)->GetState(dropout).first->data;
    CUDNN_CALL(cudnnRestoreDropoutDescriptor(dropoutDesc, CUDNNThreadEntry::ThreadLocal()->handle,
                                             dropout, state_data, stateSizeInBytes, 0));
  }

 public:
  ~DropoutImplementedByCUDNNDropoutBackward() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dxdesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dydesc));
    CUDNN_CALL(cudnnDestroyDropoutDescriptor(dropoutDesc));
  }

  std::string name() const override {
    return TruncateName(GetUniqueName("raf.op.cudnn._contrib_dropout_dx"));
  }

  void Execute(const CallValues& cv) {
    auto args = cv->args.as<raf::op::schema::DropoutDxArgs>();
    CHECK(args != nullptr);
    DLTensor* dx = cv->out;
    DLTensor* dy = args->dy;
    DLTensor* reserve_space = args->reserve_space;

    CUDNN_CALL(cudnnDropoutBackward(CUDNNThreadEntry::ThreadLocal()->handle, dropoutDesc, dydesc,
                                    dy->data, dxdesc, dx->data, reserve_space->data,
                                    reserveSpaceSizeInBytes));
  }

  void Execute(const std::vector<Value>& inputs, Value output) {
    CHECK_EQ(inputs.size(), 2);
    DLTensor* dx = output;
    DLTensor* dy = inputs[0];
    DLTensor* reserve_space = inputs[1];

    CUDNN_CALL(cudnnDropoutBackward(CUDNNThreadEntry::ThreadLocal()->handle, dropoutDesc, dydesc,
                                    dy->data, dxdesc, dx->data, reserve_space->data,
                                    reserveSpaceSizeInBytes));
  }

  static OpEnv* make(const CallValues& cv) {
    return new DropoutImplementedByCUDNNDropoutBackward(cv);
  }
};

RAF_REGISTER_DIALECT_OP(cudnn, _contrib_dropout_dx, 15);
RAF_OP_ENV_MAKER("raf.op.cudnn._contrib_dropout_dx",
                 DropoutImplementedByCUDNNDropoutBackward::make);

}  // namespace cudnn
}  // namespace op
}  // namespace raf
