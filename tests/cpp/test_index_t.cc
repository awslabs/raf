#include <gtest/gtest.h>
#include <mnm/device_api.h>
#include <mnm/types.h>
#include <cassert>
#include <iostream>

TEST(DimensionType, TextFormat) {
  using index_t = mnm::types::index_t;
  index_t a(-1);
  index_t b(-2);
  auto c = a + b;
  (a += b) += c;
  ASSERT_STREQ(std::string(a).c_str(), "-6");
  ASSERT_STREQ(std::string(b).c_str(), "-2");
  ASSERT_STREQ(std::string(c).c_str(), "-3");
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
