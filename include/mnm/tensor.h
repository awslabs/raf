#pragma once

#include <vector>

#include <tvm/runtime/ndarray.h>

#include <mnm/types.h>

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
  using TSelf = mnm::tensor::Tensor;
  using TSuper = tvm::runtime::NDArray;

  class TensorContainer;
  class Impl;

  friend TensorContainer;
  friend Impl;
  friend MNMTester;
  friend mnm::value::TensorValueNode;

 public:
  inline Tensor() = default;

  inline ~Tensor() = default;

  inline Tensor(TSelf&& other) : TSuper(other) {
  }

  inline Tensor(const TSelf& other) : TSuper(other) {
  }

  inline TSelf& operator=(TSelf&& other) {
    TSelf(std::move(other)).swap(*this);
    return *this;
  }

  inline TSelf& operator=(const TSelf& other) {
    TSelf(other).swap(*this);
    return *this;
  }

  inline void swap(TSelf& other) {
    std::swap(data_, other.data_);
  }

  // Empty strides indicates contiguous, will generate correct strides automatically
  static Tensor make(mnm::types::Context ctx,                        //
                     mnm::types::DType dtype,                        //
                     std::vector<mnm::types::index_t> shape,         //
                     std::vector<mnm::types::index_t> strides = {},  //
                     void* data = nullptr);

 public:
  // const DLTensor* operator->() const;
  using TSuper::operator->;

 private:
  Tensor(TensorContainer* data);
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
