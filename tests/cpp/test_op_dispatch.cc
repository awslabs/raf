#include <cassert>
#include <iostream>

#include <gtest/gtest.h>

#include <mnm/base.h>
#include <mnm/op.h>

using mnm::DevType;
using mnm::ir::Array;
using mnm::ir::Attrs;
using mnm::ir::Op;
using mnm::op::CallValues;
using mnm::op::OpDispatch;
using mnm::op::OpEnv;
using mnm::value::Value;

class Conv2d : public OpEnv {
 public:
  int type;
  Conv2d() = default;
  virtual ~Conv2d() = default;
  void Execute(const CallValues& call) override final {
  }
  void Execute(const std::vector<Value>& inputs, Value output) override final {
  }
};

// Implement 0 of "mnm.cpptest.conv2d"
class Conv2dX : public Conv2d {
 public:
  Conv2dX() {
    type = 0;
  }
  virtual ~Conv2dX() = default;
  static OpEnv* make(const CallValues& call) {
    return new Conv2dX();
  }
};
MNM_OP_DISPATCH("mnm.cpptest.conv2d", Conv2dX::make, DevType::kCPU(), "mklShallowNN");

// Implement 1 of "mnm.cpptest.conv2d"
class Conv2dY : public Conv2d {
 public:
  Conv2dY() {
    type = 1;
  }
  virtual ~Conv2dY() = default;
  static OpEnv* make(const CallValues& call) {
    return new Conv2dY();
  }
};
MNM_OP_DISPATCH("mnm.cpptest.conv2d", Conv2dY::make, DevType::kCPU(), "sshadow");
MNM_OP_REGISTER("mnm.cpptest.conv2d");

TEST(OpDispatch, Registry) {
  const auto* dispatch_list = OpDispatch::Get(Op::Get("mnm.cpptest.conv2d"), DevType::kCPU());
  ASSERT_EQ(dispatch_list->size(), 2);
  CallValues call;
  for (const auto e : *dispatch_list) {
    auto maker = e.maker;
    const auto* op = static_cast<Conv2d*>(maker(call));
    ASSERT_NE(op, nullptr);
    if (e.backend == "mklShallowNN") {
      ASSERT_EQ(op->type, 0);
    } else if (e.backend == "sshadow") {
      ASSERT_EQ(op->type, 1);
    } else {
      ASSERT_TRUE(false);
    }
    delete op;
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
