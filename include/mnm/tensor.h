/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file tensor.h
 * \brief Definition of MNM tensors
 */
#pragma once
#include <vector>
#include <utility>
#include "tvm/runtime/ndarray.h"
#include "tvm/runtime/packed_func.h"
#include "./device.h"

class MNMTester;

namespace mnm {
namespace value {
class TensorValueObj;
}  // namespace value
}  // namespace mnm

namespace mnm {
namespace tensor {

class Tensor : public tvm::runtime::NDArray {
  using TSelf = ::mnm::tensor::Tensor;
  using TSuper = ::tvm::runtime::NDArray;

  friend value::TensorValueObj;
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
  static Tensor make(const Device& dev, const DType& dtype, const std::vector<int64_t>& shape,
                     const std::vector<int64_t>& strides = {}, void* data = nullptr);

  static Tensor FromDLPack(DLManagedTensor* tensor);

  DLManagedTensor* ToDLPack() const;

  void CopyTo(const Tensor& other) const;

  void CopyTo(DLTensor* other) const;

  NDArray CopyTo(const Device& dev) const;

 public:
  using TSuper::operator->;

  class Impl;
  class TensorContainer;

  using ContainerType = TensorContainer;

  Tensor(tvm::runtime::ObjectPtr<tvm::runtime::Object> data);  // NOLINT(runtime/explicit)
};

}  // namespace tensor
}  // namespace mnm
