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

#include <cstdlib>
#include <functional>
#include <numeric>

#include <gtest/gtest.h>

#include <dlpack/dlpack.h>

#include <mnm/device.h>
#include <mnm/tensor.h>

using mnm::Device;
using mnm::DevType;
using mnm::DType;
using mnm::DTypeCode;
using mnm::tensor::Tensor;

class MNMTester {
 public:
  static int use_count(const Tensor& tensor) {
    return tensor.use_count();
  }
  static std::vector<int64_t> RandomShape(int ndim) {
    std::vector<int64_t> result(ndim);
    for (int i = 0; i < ndim; ++i) {
      result[i] = std::rand() % 10;
    }
    return result;
  }
  static std::vector<int64_t> CalcStrides(const std::vector<int64_t>& shape) {
    int ndim = shape.size();
    std::vector<int64_t> result(ndim);
    for (int i = 0; i < ndim; ++i) {
      result[i] = std::accumulate(shape.begin() + i + 1, shape.end(), 1, std::multiplies<int>());
    }
    return result;
  }
};

TEST(Tensor, make_no_strides_no_data) {
  const int ndim = 3;
  Device dev(DevType::kCPU(), 0);        // cpu(0)
  DType dtype(DTypeCode::kFloat(), 16);  // float16
  std::vector<int64_t> shape = MNMTester::RandomShape(ndim);
  std::vector<int64_t> strides = MNMTester::CalcStrides(shape);
  // call make
  Tensor tensor = Tensor::make(dev, dtype, shape);
  // check ndim
  ASSERT_EQ(tensor->ndim, ndim);
  // check data
  ASSERT_EQ(tensor->data, nullptr);
  // check dev
  ASSERT_EQ(tensor->device.device_type, kDLCPU);
  ASSERT_EQ(tensor->device.device_id, 0);
  // check dtype
  ASSERT_EQ(tensor->dtype.code, kDLFloat);
  ASSERT_EQ(tensor->dtype.bits, 16);
  ASSERT_EQ(tensor->dtype.lanes, 1);
  // check shape
  ASSERT_EQ(shape, std::vector<int64_t>(tensor->shape, tensor->shape + ndim));
  // check strides
  ASSERT_EQ(strides, std::vector<int64_t>(tensor->strides, tensor->strides + ndim));
  // check byte_offset
  ASSERT_EQ(tensor->byte_offset, 0);
  // check use_count
  ASSERT_EQ(MNMTester::use_count(tensor), 1);
}

TEST(Tensor, make_no_strides) {
  const int ndim = 3;
  void* data = (void*)0x0001;
  Device dev(DevType::kCPU(), 0);        // cpu(0)
  DType dtype(DTypeCode::kFloat(), 16);  // float16
  std::vector<int64_t> shape = MNMTester::RandomShape(ndim);
  std::vector<int64_t> strides = MNMTester::CalcStrides(shape);
  // call make
  Tensor tensor = Tensor::make(dev, dtype, shape, {}, data);
  // check ndim
  ASSERT_EQ(tensor->ndim, ndim);
  // check data
  ASSERT_EQ(tensor->data, data);
  // check dev
  ASSERT_EQ(tensor->device.device_type, kDLCPU);
  ASSERT_EQ(tensor->device.device_id, 0);
  // check dtype
  ASSERT_EQ(tensor->dtype.code, kDLFloat);
  ASSERT_EQ(tensor->dtype.bits, 16);
  ASSERT_EQ(tensor->dtype.lanes, 1);
  // check shape
  ASSERT_EQ(shape, std::vector<int64_t>(tensor->shape, tensor->shape + ndim));
  // check strides
  ASSERT_EQ(strides, std::vector<int64_t>(tensor->strides, tensor->strides + ndim));
  // check byte_offset
  ASSERT_EQ(tensor->byte_offset, 0);
  // check use_count
  ASSERT_EQ(MNMTester::use_count(tensor), 1);
}

TEST(Tensor, make_given_all_fields) {
  const int ndim = 3;
  void* data = (void*)0x0001;
  Device dev(DevType::kCPU(), 0);        // cpu(0)
  DType dtype(DTypeCode::kFloat(), 16);  // float16
  std::vector<int64_t> shape = MNMTester::RandomShape(ndim);
  std::vector<int64_t> strides = MNMTester::RandomShape(ndim);
  // call make
  Tensor tensor = Tensor::make(dev, dtype, shape, strides, data);
  // check ndim
  ASSERT_EQ(tensor->ndim, ndim);
  // check data
  ASSERT_EQ(tensor->data, data);
  // check dev
  ASSERT_EQ(tensor->device.device_type, kDLCPU);
  ASSERT_EQ(tensor->device.device_id, 0);
  // check dtype
  ASSERT_EQ(tensor->dtype.code, kDLFloat);
  ASSERT_EQ(tensor->dtype.bits, 16);
  ASSERT_EQ(tensor->dtype.lanes, 1);
  // check shape
  ASSERT_EQ(shape, std::vector<int64_t>(tensor->shape, tensor->shape + ndim));
  // check strides
  ASSERT_EQ(strides, std::vector<int64_t>(tensor->strides, tensor->strides + ndim));
  // check byte_offset
  ASSERT_EQ(tensor->byte_offset, 0);
  // check use_count
  ASSERT_EQ(MNMTester::use_count(tensor), 1);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
