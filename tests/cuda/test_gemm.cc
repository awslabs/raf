#include <algorithm>
#include <cassert>
#include <iostream>

#include <cublas.h>
#include <cuda.h>
#include <gtest/gtest.h>

#include <mnm/base.h>
#include <mnm/op.h>
#include <mnm/registry.h>
#include <mnm/value.h>

using mnm::Context;
using mnm::DevType;
using mnm::DType;
using mnm::DTypeCode;
using mnm::op::OpDispatch;
using mnm::op::OpEnv;
using mnm::registry::Registry;
using mnm::rly::Array;
using mnm::rly::Attrs;
using mnm::rly::Integer;
using mnm::value::TensorValue;
using mnm::value::Value;

TEST(cuBlas, Gemm) {
  const auto* dispatch_list = OpDispatch::Get("mnm.op.linear", DevType::kCUDA());
  ASSERT_EQ(dispatch_list->size(), 1);

  Context ctx(DevType::kCUDA(), 0);
  DType dt(DTypeCode::kFloat(), 32, 1);

  std::vector<int64_t> a_dims{32, 128};
  void* a_data = nullptr;
  cudaMalloc(&a_data, 32 * 128 * sizeof(float));
  auto data = TensorValue::Assemble(ctx, dt, a_dims, {}, a_data);

  std::vector<int64_t> b_dims{512, 128};
  void* b_data = nullptr;
  cudaMalloc(&b_data, 128 * 512 * sizeof(float));
  auto weight = TensorValue::Assemble(ctx, dt, b_dims, {}, b_data);

  std::vector<int64_t> o_dims{32, 512};
  void* o_data = nullptr;
  cudaMalloc(&o_data, 32 * 512 * sizeof(float));
  auto output = TensorValue::Assemble(ctx, dt, o_dims, {}, o_data);

  Array<Value> args{data, weight, output};
  Attrs attrs;
  for (const auto& e : *dispatch_list) {
    OpEnv* op = (OpEnv*)e.second(args, attrs);
    op->Execute(args, attrs);
    delete op;
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
