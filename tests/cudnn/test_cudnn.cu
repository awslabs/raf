#include <cudnn.h>
#include <gtest/gtest.h>

#define N 1000

int a[N], b[N], c[N], ref[N];

TEST(CuDNN, CreateDestroyHandle) {
  cudnnHandle_t handle;
  cudnnCreate(&handle);
  cudnnDestroy(handle);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
