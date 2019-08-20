#include <vector>

#include <mnm/registry.h>
#include <mnm/tensor.h>

#include "../common/shape_utils.h"

namespace mnm {
namespace tensor {

using common::shape_utils::Shape2Strides;

class Tensor::TensorContainer : public tvm::runtime::NDArray::Container {
 public:
  // DLTensor dl_tensor;
  using Container::dl_tensor;
  // void* manager_ctx = nullptr;
  using Container::manager_ctx;
  // void (*deleter)(Container* self) = nullptr;
  using Container::deleter;
  // int32_t array_type_code_ = 0;
  using Container::array_type_code_;
  //  std::atomic<int> ref_counter_ = 0;
  using Container::ref_counter_;
  // std::vector<int64_t> shape_;
  using Container::shape_;
  // An extra field
  std::vector<int64_t> strides_;

  TensorContainer() = delete;

  TensorContainer(void (*del)(Container* self)) {
    deleter = del;
    array_type_code_ = kArrayTypeCode;
  }

  void CheckTypeCode() {
    CHECK_EQ(array_type_code_, kArrayTypeCode)
        << "InternalError: type code error " << array_type_code_;
  }
};

class Tensor::Impl {
 public:
  static void DefaultDeleter(Container* super_ptr) {
    TensorContainer* ptr = static_cast<TensorContainer*>(super_ptr);
    if (ptr->manager_ctx != nullptr) {
      // View of other tensors
      static_cast<TSuper::Container*>(ptr->manager_ctx)->DecRef();
    } else {
      CHECK_EQ(ptr->array_type_code_, kArrayTypeCode);
      // Memory is not owned by MNM tensor, so do nothing
    }
    delete ptr;
  }

  static void NumpyArrayDeleter(Container* super_ptr) {
    static const auto& deleter = registry::GetPackedFunc("mnm._numpy_array_deleter");
    TensorContainer* ptr = static_cast<TensorContainer*>(super_ptr);
    CHECK(ptr->manager_ctx != nullptr);
    deleter(ptr->manager_ctx);
    delete ptr;
  }

  static Tensor Make(const Context& ctx, const DType& dtype, const std::vector<int64_t>& shape,
                     const std::vector<int64_t>& strides, void* data) {
    if (!strides.empty()) {
      CHECK_EQ(shape.size(), strides.size());
    }
    TensorContainer* container = new TensorContainer(DefaultDeleter);
    Tensor ret(container);
    container->shape_ = shape;
    container->strides_ = !strides.empty() ? strides : Shape2Strides<int64_t>(container->shape_);
    container->dl_tensor.data = data;
    container->dl_tensor.ctx = ctx;
    container->dl_tensor.ndim = shape.size();
    container->dl_tensor.dtype = dtype;
    container->dl_tensor.shape = dmlc::BeginPtr(container->shape_);
    container->dl_tensor.strides = dmlc::BeginPtr(container->strides_);
    container->dl_tensor.byte_offset = 0;
    return ret;
  }

  static void MarkNumpy(Tensor tensor, void* manager_ctx) {
    tensor.data_->manager_ctx = manager_ctx;
    tensor.data_->deleter = NumpyArrayDeleter;
  }
};

Tensor::Tensor(tvm::runtime::NDArray::Container* data) : TSuper(data) {
}

Tensor::Tensor(const tvm::runtime::NDArray& other) : TSuper(other) {
  static_cast<Tensor::TensorContainer*>(data_)->CheckTypeCode();
}

int Tensor::array_type_code() const {
  return data_ == nullptr ? -1 : static_cast<Tensor::TensorContainer*>(data_)->array_type_code_;
}

Tensor Tensor::make(const Context& ctx, const DType& dtype, const std::vector<int64_t>& shape,
                    const std::vector<int64_t>& strides, void* data) {
  return Tensor::Impl::Make(ctx, dtype, shape, strides, data);
}

MNM_REGISTER_GLOBAL("mnm.tensor.MarkNumpy").set_body_typed(Tensor::Impl::MarkNumpy);

}  // namespace tensor
}  // namespace mnm
