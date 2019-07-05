#include <algorithm>
#include <cassert>
#include <iostream>

#include <cuda.h>
#include <cudnn.h>
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

TEST(CuDNN, RegisterConvCudnn) {
  const auto* dispatch_list = OpDispatch::Get("mnm.conv2d", DevType::kCUDA());
  const auto* MakeNode = Registry::Get("make._Node");
  ASSERT_EQ(dispatch_list->size(), 1);

  Context ctx(DevType::kCUDA(), 0);
  DType dt(DTypeCode::kFloat(), 32, 1);

  std::vector<int64_t> a_dims{3, 128, 128};
  void* a_data = nullptr;
  cudaMalloc(&a_data, 3 * 1440 * 900 * sizeof(float));
  auto img = TensorValue::Assemble(ctx, dt, a_dims, {}, a_data);

  std::vector<int64_t> b_dims{512, 3, 3, 3};
  void* b_data = nullptr;
  cudaMalloc(&b_data, 512 * 3 * 3 * 3 * sizeof(float));
  auto fil = TensorValue::Assemble(ctx, dt, b_dims, {}, b_data);

  Array<Value> args{img, fil};
  Attrs attrs = (*MakeNode)("mnm.attrs.Conv2DAttrs",           //
                            "stride", Array<Integer>{1, 1},    //
                            "padding", Array<Integer>{0, 0},   //
                            "dilation", Array<Integer>{1, 1},  //
                            "groups", Integer{1});
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
