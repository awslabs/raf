/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/impl/tensor.cc
 * \brief RAF Tensor underlying implementation
 */

#include <raf/registry.h>
#include <raf/tensor.h>
#include <vector>
#include "raf/device_api.h"
#include "../common/shape_utils.h"

namespace raf {
namespace tensor {

using common::shape_utils::GetShape;
using common::shape_utils::IsCompact;
using common::shape_utils::Shape2Strides;

class Tensor::TensorContainer : public ir::NDArray::Container {
 public:
  // DLTensor dl_tensor;
  using Container::dl_tensor;
  // void* manager_ctx = nullptr;
  using Container::manager_ctx;
  // void (*deleter)(Container* self) = nullptr;
  using Container::deleter_;
  // std::atomic<int> ref_counter_ = 0;
  using Container::ref_counter_;
  // std::vector<int64_t> shape_;
  using Container::shape_;
  // An extra field
  std::vector<int64_t> strides_;

  TensorContainer() : ir::NDArray::Container() {
    type_index_ = TensorContainer::RuntimeTypeIndex();
  }

  static constexpr const uint32_t _type_index = tvm::TypeIndex::kDynamic;
  static constexpr const char* _type_key = "raf.tensor.Tensor";
  TVM_DECLARE_FINAL_OBJECT_INFO(TensorContainer, ir::NDArray::Container);
};

class Tensor::Impl {
 public:
  static void DefaultDeleter(ir::Object* super_ptr) {
    TensorContainer* ptr = static_cast<TensorContainer*>(super_ptr);
    if (ptr->manager_ctx != nullptr) {
      // View of other tensors
      static_cast<TSuper::Container*>(ptr->manager_ctx)->DecRef();
    } else {
      // Memory is not owned by RAF tensor, so do nothing
    }
    delete ptr;
  }

  static void NumpyArrayDeleter(ir::Object* super_ptr) {
    TensorContainer* ptr = static_cast<TensorContainer*>(super_ptr);
    CHECK(ptr->manager_ctx != nullptr);
    static const auto& deleter = registry::GetPackedFunc("raf._numpy_array_deleter");
    deleter(ptr->manager_ctx);
    delete ptr;
  }

  static void ToDLPackDeleter(DLManagedTensor* tensor) {
    static_cast<NDArray::Container*>(tensor->manager_ctx)->DecRef();
    delete tensor;
  }

  static void FromDLPackDeleter(ir::Object* super_ptr) {
    auto* ptr = static_cast<NDArray::Container*>(super_ptr);
    DLManagedTensor* tensor = static_cast<DLManagedTensor*>(ptr->manager_ctx);
    if (tensor->deleter != nullptr) {
      (*tensor->deleter)(tensor);
    }
    delete super_ptr;
  }

  static Tensor Make(const Device& dev, const DType& dtype, const std::vector<int64_t>& shape,
                     const std::vector<int64_t>& strides, void* data) {
    if (!strides.empty()) {
      CHECK_EQ(shape.size(), strides.size());
    }
    TensorContainer* container = new TensorContainer();
    container->SetDeleter(DefaultDeleter);
    Tensor ret(ir::GetObjectPtr<ir::Object>(container));
    container->shape_ = shape;
    container->strides_ = !strides.empty() ? strides : Shape2Strides<int64_t>(shape);
    container->dl_tensor.data = data;
    container->dl_tensor.device = dev;
    container->dl_tensor.ndim = shape.size();
    container->dl_tensor.dtype = dtype;
    container->dl_tensor.shape = const_cast<int64_t*>(container->shape_.data());
    container->dl_tensor.strides = dmlc::BeginPtr(container->strides_);
    container->dl_tensor.byte_offset = 0;
    return ret;
  }

  static Tensor FromDLPack(DLManagedTensor* tensor) {
    TensorContainer* container = new TensorContainer();
    container->SetDeleter(FromDLPackDeleter);
    Tensor ret(ir::GetObjectPtr<ir::Object>(container));
    container->manager_ctx = tensor;
    container->dl_tensor = tensor->dl_tensor;
    std::vector<int64_t> shape(tensor->dl_tensor.shape,
                               tensor->dl_tensor.shape + tensor->dl_tensor.ndim);
    container->strides_ = Shape2Strides<int64_t>(shape);
    container->shape_ = std::move(shape);
    container->dl_tensor.shape = const_cast<int64_t*>(container->shape_.data());
    container->dl_tensor.strides = dmlc::BeginPtr(container->strides_);
    return ret;
  }

  static void MarkNumpy(Tensor tensor, void* manager_ctx) {
    tensor.get_mutable()->manager_ctx = manager_ctx;
    tensor.get_mutable()->SetDeleter(NumpyArrayDeleter);
  }

  static Tensor CreateView(const Tensor& self,
                           const std::vector<int64_t>& shape,    //
                           const std::vector<int64_t>& strides,  //
                           void* data) {
    if (!strides.empty()) {
      CHECK_EQ(shape.size(), strides.size());
    }
    TensorContainer* container = new TensorContainer();
    container->SetDeleter(DefaultDeleter);
    Tensor ret(ir::GetObjectPtr<ir::Object>(container));
    container->shape_ = shape;
    container->strides_ = !strides.empty() ? strides : Shape2Strides<int64_t>(shape);
    container->dl_tensor.device = self->device;
    container->dl_tensor.ndim = container->shape_.size();
    container->dl_tensor.dtype = self->dtype;
    container->dl_tensor.shape = const_cast<int64_t*>(container->shape_.data());
    container->dl_tensor.strides = dmlc::BeginPtr(container->strides_);
    container->dl_tensor.byte_offset = 0;
    self.get_mutable()->IncRef();
    container->manager_ctx = self.get_mutable();
    container->dl_tensor.data = data ? data : self->data;
    return ret;
  }

  /*!
   * \brief Function to copy data from one array to another.
   * \param from The source array.
   * \param to The target array.
   */
  static void CopyFromTo(const DLTensor* from, DLTensor* to) {
    size_t from_size = tvm::runtime::GetDataSize(*from);
    size_t to_size = tvm::runtime::GetDataSize(*to);
    ICHECK_EQ(from_size, to_size) << "CopyFromTo: The size must exactly match";

    ICHECK(from->device.device_type == to->device.device_type ||
           from->device.device_type == kDLCPU || to->device.device_type == kDLCPU ||
           from->device.device_type == kDLCUDAHost || to->device.device_type == kDLCUDAHost)
        << "Can not copy across different device types directly. From device type: "
        << from->device.device_type << " to device type: " << to->device.device_type;

    // Use the device that is NOT CPU (or CUDA host) to get the correct device api manager.
    Device dev = from->device;
    if (from->device.device_type == kDLCPU || from->device.device_type == kDLCUDAHost) {
      dev = to->device;
    }
    auto dapi = device_api::DeviceAPI::Get(dev->device_type);
    dapi->CopyDataFromTo(const_cast<DLTensor*>(from), to, dapi->GetStream());
  }
};

Tensor::Tensor(ir::ObjectPtr<ir::Object> data) : TSuper(data) {
}

Tensor::Tensor(const ir::NDArray& other) : TSuper(other) {
}

Tensor Tensor::CreateView(const std::vector<int64_t>& shape, const std::vector<int64_t>& strides,
                          void* data) const {
  return Tensor::Impl::CreateView(*this, shape, strides, data);
}

Tensor Tensor::make(const Device& dev, const DType& dtype, const std::vector<int64_t>& shape,
                    const std::vector<int64_t>& strides, void* data) {
  return Tensor::Impl::Make(dev, dtype, shape, strides, data);
}

Tensor Tensor::FromDLPack(DLManagedTensor* tensor) {
  return Tensor::Impl::FromDLPack(tensor);
}

void Tensor::CopyTo(const Tensor& other) const {
  ICHECK(data_ != nullptr);
  ICHECK(other.data_ != nullptr);
  Tensor::Impl::CopyFromTo(&(get_mutable()->dl_tensor), &(other.get_mutable()->dl_tensor));
}

void Tensor::CopyTo(DLTensor* other) const {
  ICHECK(data_ != nullptr);
  Tensor::Impl::CopyFromTo(&(get_mutable()->dl_tensor), other);
}

NDArray Tensor::CopyTo(const Device& dev) const {
  // TODO: Support stream.
  ICHECK(data_ != nullptr);
  const DLTensor* dptr = operator->();
  NDArray ret =
      Empty(tvm::runtime::ShapeTuple(dptr->shape, dptr->shape + dptr->ndim), dptr->dtype, dev);
  this->CopyTo(ret);
  return ret;
}

DLManagedTensor* Tensor::ToDLPack() const {
  DLManagedTensor* ret = new DLManagedTensor();
  ret->deleter = Tensor::Impl::ToDLPackDeleter;
  ret->manager_ctx = get_mutable();
  ret->dl_tensor = get_mutable()->dl_tensor;
  if (IsCompact(get_mutable()->dl_tensor)) {
    ret->dl_tensor.strides = nullptr;
  }
  get_mutable()->IncRef();
  return ret;
}

TVM_REGISTER_OBJECT_TYPE(Tensor::TensorContainer);

RAF_REGISTER_GLOBAL("raf.tensor.MarkNumpy").set_body_typed(Tensor::Impl::MarkNumpy);

}  // namespace tensor
}  // namespace raf
