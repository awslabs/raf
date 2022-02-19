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
