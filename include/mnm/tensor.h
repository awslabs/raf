/*!
 * Copyright (c) 2019 by Contributors
 * \file tensor.h
 * \brief Definition of MNM tensors
 */
#pragma once
#include <vector>
#include <utility>
#include "tvm/runtime/ndarray.h"
#include "tvm/runtime/packed_func.h"
#include "mnm/base.h"

class MNMTester;

namespace mnm {
namespace value {
class TensorValueNode;
}  // namespace value
}  // namespace mnm

namespace mnm {
namespace tensor {

constexpr int kArrayTypeCode = 1;

class Tensor : private tvm::runtime::NDArray {
  using TSelf = ::mnm::tensor::Tensor;
  using TSuper = ::tvm::runtime::NDArray;

  friend value::TensorValueNode;
  friend MNMTester;
  friend ::tvm::runtime::TVMPODValue_;

 public:
  Tensor() = default;

  ~Tensor() = default;

  Tensor(TSelf&& other) : TSuper(other) {  // NOLINT(runtime/explicit)
  }

  Tensor(const TSelf& other) : TSuper(other) {  // NOLINT(runtime/explicit)
  }

  Tensor(const TSuper& other);  // NOLINT(runtime/explicit)

  TSelf CreateView(const std::vector<int64_t>& shape = {}, const std::vector<int64_t>& strides = {},
                   void* data = nullptr) const;

  TSelf& operator=(TSelf&& other) {
    TSelf(std::move(other)).swap(*this);
    return *this;
  }

  TSelf& operator=(const TSelf& other) {
    TSelf(other).swap(*this);
    return *this;
  }

  void swap(TSelf& other) {
    std::swap(data_, other.data_);
  }

  // Empty strides indicates contiguous, will generate correct strides automatically
  static Tensor make(const Context& ctx, const DType& dtype, const std::vector<int64_t>& shape,
                     const std::vector<int64_t>& strides = {}, void* data = nullptr);

  static Tensor FromDLPack(DLManagedTensor* tensor);

  DLManagedTensor* ToDLPack() const;

 public:
  // const DLTensor* operator->() const;
  using TSuper::operator->;

  class Impl;
  class TensorContainer;

 private:
  Tensor(tvm::runtime::NDArray::Container* data);  // NOLINT(runtime/explicit)

  int array_type_code() const;
};

}  // namespace tensor
}  // namespace mnm

namespace tvm {
namespace runtime {
template <>
struct array_type_info<::mnm::tensor::Tensor> {
  static const int code = ::mnm::tensor::kArrayTypeCode;
};
}  // namespace runtime
}  // namespace tvm
