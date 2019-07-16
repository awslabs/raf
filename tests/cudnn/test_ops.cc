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
using mnm::rly::Float;
using mnm::rly::Integer;
using mnm::value::TensorValue;
using mnm::value::Value;

Value MakeTensor(const std::vector<int64_t>& shape, bool allocate) {
  Context ctx(DevType::kCUDA(), 0);
  DType dt(DTypeCode::kFloat(), 32, 1);
  void* data = nullptr;
  int64_t size = sizeof(float);
  for (int i = 0; i < shape.size(); ++i) size *= shape[i];
  cudaMalloc(&data, size);
  return TensorValue::Assemble(ctx, dt, shape, {}, data);
}

TEST(CUDNN, Convolution) {
  const auto* dispatch_list = OpDispatch::Get("mnm.op.conv2d", DevType::kCUDA());
  ASSERT_EQ(dispatch_list->size(), 2);

  auto img = MakeTensor({3, 128, 128}, true);
  auto fil = MakeTensor({512, 3, 3, 3}, true);
  auto out = MakeTensor({1, 512, 126, 126}, false);

  const auto* MakeNode = Registry::Get("make._Node");
  Attrs attrs = (*MakeNode)("mnm.attrs.Conv2DAttrs",           //
                            "stride", Array<Integer>{1, 1},    //
                            "padding", Array<Integer>{0, 0},   //
                            "dilation", Array<Integer>{1, 1},  //
                            "groups", Integer{1});

  Array<Value> args{img, fil, out};

  for (const auto& e : *dispatch_list) {
    std::cout << "Construct: " << e.first->name << "\n";
    OpEnv* op = (OpEnv*)e.second(args, attrs);
    op->Execute(args, attrs);
    delete op;
  }

  for (const auto& e : *dispatch_list) {
    std::cout << "Construct: " << e.first->name << "\n";
    OpEnv* op = (OpEnv*)e.second(args, attrs);
    op->Execute(args, attrs);
    delete op;
  }
}

TEST(CUDNN, Identical) {
  const auto* relu_list = OpDispatch::Get("mnm.op.relu", DevType::kCUDA());
  Attrs empty;
  auto img = MakeTensor({8, 8, 8, 8}, true);
  auto out = MakeTensor({8, 8, 8, 8}, false);
  Array<Value> args{img, out};

  for (const auto& e : *relu_list) {
    OpEnv* op = (OpEnv*)e.second(args, empty);
    op->Execute(args, empty);
    delete op;
  }

  const auto* MakeNode = Registry::Get("make._Node");
  Attrs attrs = (*MakeNode)("mnm.attrs.DropoutAttrs", "dropout", Float(0.5));
  const auto* dropout_list = OpDispatch::Get("mnm.op.dropout", DevType::kCUDA());

  for (const auto& e : *dropout_list) {
    OpEnv* op = (OpEnv*)e.second(args, attrs);
    op->Execute(args, attrs);
    delete op;
  }

  for (const auto& e : *dropout_list) {
    OpEnv* op = (OpEnv*)e.second(args, attrs);
    op->Execute(args, attrs);
    delete op;
  }
}

TEST(CUDNN, MaxPool2D) {
  const auto* dispatch_list = OpDispatch::Get("mnm.op.max_pool2d", DevType::kCUDA());
  ASSERT_EQ(dispatch_list->size(), 1);

  auto img = MakeTensor({128, 128}, true);
  auto out = MakeTensor({126, 126}, false);

  const auto* MakeNode = Registry::Get("make._Node");
  Attrs attrs = (*MakeNode)("mnm.attrs.MaxPoolAttrs", "kernel_size", Array<Integer>{3, 3}, "stride",
                            Array<Integer>{1, 1}, "padding", Array<Integer>{0, 0}, "dilation",
                            Array<Integer>{1, 1}, "ceil_mode", false);

  Array<Value> args{img, out};

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
