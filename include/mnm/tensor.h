#pragma once

#include <mnm/memory_pool.h>
#include <mnm/types.h>
#include <tvm/runtime/ndarray.h>

namespace mnm {
namespace tensor {
class Tensor;
}  // namespace tensor
}  // namespace mnm

namespace tvm {
namespace runtime {
template <>
struct array_type_info<mnm::tensor::Tensor> {
  static const int code = 1;
};
}  // namespace runtime
}  // namespace tvm

namespace mnm {
namespace tensor {

class Tensor : public tvm::runtime::NDArray {
  using TSelf = mnm::tensor::Tensor;
  using TSuper = tvm::runtime::NDArray;

 public:
  class Container : private TSuper::Container {
    class Impl;
    friend class mnm::tensor::Tensor;
    friend class mnm::tensor::Tensor::Container::Impl;

   public:
    Container() : TSuper::Container() {
      array_type_code_ = tvm::runtime::array_type_info<TSelf>::code;
    }

    void SwitchFromSuper() {
      if (array_type_code_ == tvm::runtime::array_type_info<TSelf>::code) {
        return;
      }
      CHECK_EQ(array_type_code_, 0)
          << "InternalError: Cannot switch from type code " << array_type_code_;
      array_type_code_ = tvm::runtime::array_type_info<TSelf>::code;
      CHECK(dl_tensor.strides == nullptr) << "InternalError: TVM's NDArray should not use strides";
      strides_.clear();
      strides_.shrink_to_fit();
      memory_chunk_ = nullptr;
    }

   private:
    using TSuper::Container::array_type_code_;
    using TSuper::Container::deleter;
    using TSuper::Container::dl_tensor;
    using TSuper::Container::manager_ctx;
    using TSuper::Container::shape_;

    // For dlpack compatibility, we broke the strongly-typed members
    std::vector<int64_t> strides_{};
    // Here NDArray doesn't have ownership over memory chunk, instead it is managed by MemoryManager
    mnm::memory_pool::MemoryChunk* memory_chunk_{nullptr};
  };

 public:
  // Constructors from TSelf
  Tensor() : TSuper() {
  }
  Tensor(TSelf&& other) : TSuper(other) {
  }
  Tensor(const TSelf& other) : TSuper(other) {
  }
  Tensor(std::nullptr_t null) : TSuper(nullptr) {
  }
  explicit Tensor(TSelf::Container* data) : TSuper(data) {
  }
  explicit Tensor(TSuper::Container* data) : TSuper(data) {
    static_cast<TSelf::Container*>(data_)->SwitchFromSuper();
  }
  // Destructor: call superclass's destructor, and everything will just be fine.
  ~Tensor() = default;
  // Swap and copies
  void swap(TSelf& other) {
    std::swap(data_, other.data_);
  }
  TSelf& operator=(const TSelf& other) {
    TSelf(other).swap(*this);
    return *this;
  }
  TSelf& operator=(TSelf&& other) {
    TSelf(std::move(other)).swap(*this);
    return *this;
  }
  // shared_ptr infra
  bool defined() const {
    return data_ != nullptr;
  }
  bool same_as(const TSelf& other) const {
    return data_ == other.data_;
  }
  bool same_as(const TSuper& other) const {
    return data_ == static_cast<const TSelf&>(other).data_;
  }
  void reset() {
    TSuper::reset();
  }
  int use_count() const {
    return TSuper::use_count();
  }
  // Creation
  static TSelf Empty(std::vector<int64_t> shape, mnm::types::DType dtype, mnm::types::Context ctx);
  // TODO(@junrushao1994): Serialization
  bool Load(dmlc::Stream* stream);
  void Save(dmlc::Stream* stream) const;
  // TSuper compatibility layer
  static TSelf MoveFromSuper(TSuper&& other) {
    TSelf ret(other);
    static_cast<TSelf::Container*>(ret.data_)->SwitchFromSuper();
    return ret;
  }
  static TSelf CreateFromSuper(const TSuper& other) {
    TSelf ret(other);
    static_cast<TSelf::Container*>(ret.data_)->SwitchFromSuper();
    return ret;
  }
  static TSelf CopyFromSuper(const TSuper& other);  // TODO(@were): copy
  // DLTensor compatibility layer: [caution] it does not transfer/share ownership
  static TSelf MoveFromDLTensor(DLTensor&& other);
  static TSelf CreateFromDLTensor(const DLTensor& other);
  static TSelf CopyFromDLTensor(const DLTensor& other);
  DLTensor CreateToDLTensor() const;   // The DLTensor created does not retain any ownership
  DLTensor MoveToDLTensor() = delete;  // not supported: will make ownership look weird
  DLTensor CopyToDLTensor() = delete;  // not supported: will make ownership look weird
  // DLManagedTensor compatibility layer
  static TSelf MoveFromDLManagedTensor(DLManagedTensor&& other);
  static TSelf CreateFromDLManagedTensor(const DLManagedTensor& other);
  static TSelf CopyFromDLManagedTensor(const DLManagedTensor& other);  // TODO(@were)
  DLManagedTensor MoveToDLManagedTensor();
  DLManagedTensor CreateToDLManagedTensor() const;
  DLManagedTensor CopyToDLManagedTensor() const;

 private:
  // Only for internal use, external users should explicitly call CreateFromSuper or MoveFromSuper
  // to enable extra checkings, and avoid throwing errors inside constructors.
  Tensor(TSuper&& other) : TSuper(other) {
  }
  Tensor(const TSuper& other) : TSuper(other) {
  }
};

}  // namespace tensor
}  // namespace mnm
