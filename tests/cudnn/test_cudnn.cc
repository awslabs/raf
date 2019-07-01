#include <iostream>
#include <cuda.h>
#include <cudnn.h>
#include <gtest/gtest.h>
#include <sys/time.h>

TEST(CUDNN, CreateDestroyHandle) {
  cudnnHandle_t handle;

  ASSERT_EQ(cudnnCreate(&handle), CUDNN_STATUS_SUCCESS);
  ASSERT_EQ(cudnnDestroy(handle), CUDNN_STATUS_SUCCESS);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
