#include <cassert>
#include <iostream>

#include <gtest/gtest.h>

#include <mnm/op.h>
#include <mnm/types.h>

MNM_REGISTER_OP_BACKEND("cuShallowNN")  //
    .set_device(mnm::types::DeviceType::kGPU())
    .set_priority(10);

MNM_REGISTER_OP_BACKEND("mklShallowNN")  //
    .set_device(mnm::types::DeviceType::kCPU())
    .set_priority(20);

TEST(OpBackend, Registry) {
  using mnm::op::OpBackend;
  OpBackend* cuSNN = OpBackend::Get("cuShallowNN");
  ASSERT_EQ(cuSNN->name, std::string("cuShallowNN"));
  ASSERT_EQ(cuSNN->priority, 10);
  ASSERT_STREQ(cuSNN->device.c_str(), "gpu");
  OpBackend* mklSNN = OpBackend::Get("mklShallowNN");
  ASSERT_EQ(mklSNN->name, std::string("mklShallowNN"));
  ASSERT_EQ(mklSNN->priority, 20);
  ASSERT_STREQ(mklSNN->device.c_str(), "cpu");
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
