#include <algorithm>
#include <cassert>
#include <iostream>

#include <cublas.h>
#include <cuda.h>
#include <gtest/gtest.h>

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
