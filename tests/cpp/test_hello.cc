#include <gtest/gtest.h>
#include <mnm/device_api.h>
#include <mnm/test.h>
#include <mnm/types.h>
#include <cassert>
#include <iostream>

TEST(Hello, Hello) {
  print_hello();
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
