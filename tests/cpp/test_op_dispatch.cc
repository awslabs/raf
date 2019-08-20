#include <cassert>
#include <iostream>

#include <gtest/gtest.h>

#include <mnm/base.h>
#include <mnm/op.h>

using mnm::DevType;
using mnm::ir::Array;
using mnm::ir::Attrs;
using mnm::op::OpDispatch;
using mnm::op::OpEnv;
using mnm::value::Value;

class Conv2d : public OpEnv {
 public:
  int type = -1;

  Conv2d() = default;
  virtual ~Conv2d() = default;

  void PreAlloc(Array<Value> args, Attrs attrs) {
  }

  void Execute(Array<Value> args, Value output, Attrs attrs) override final {
  }
};

// Implement 0 of "mnm.conv2d"
class Conv2dX : public Conv2d {
 public:
  Conv2dX() {
    type = 0;
  }
  virtual ~Conv2dX() = default;
  static OpEnv* make(Array<Value> args, Value output, Attrs attrs) {
    std::unique_ptr<Conv2dX> ret(new Conv2dX());
    ret->PreAlloc(args, attrs);
    return ret.release();
  }
};

MNM_REGISTER_OP_DISPATCH("mnm.op.conv2d", DevType::kCPU(), "mklShallowNN", Conv2dX::make);

// Implement 1 of "mnm.conv2d"
class Conv2dY : public Conv2d {
 public:
  Conv2dY() {
    type = 1;
  }
  virtual ~Conv2dY() = default;
  static OpEnv* make(Array<Value> args, Value output, Attrs attrs) {
    std::unique_ptr<Conv2dY> ret(new Conv2dY());
    ret->PreAlloc(args, attrs);
    return ret.release();
  }
};

MNM_REGISTER_OP_DISPATCH("mnm.op.conv2d", DevType::kCPU(), "sshadow", Conv2dY::make);

TEST(OpDispatch, Registry) {
  const auto* dispatch_list = OpDispatch::Get("mnm.op.conv2d", DevType::kCPU());
  ASSERT_EQ(dispatch_list->size(), 2);
  Array<Value> args;
  Value output;
  Attrs attrs;
  for (const auto& e : *dispatch_list) {
    const auto* op = static_cast<Conv2d*>(e.second(args, output, attrs));
    ASSERT_NE(op, nullptr);
    if (e.first->name == "mklShallowNN") {
      ASSERT_EQ(op->type, 0);
    } else if (e.first->name == "sshadow") {
      ASSERT_EQ(op->type, 1);
    } else {
      ASSERT_TRUE(false);
    }
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
