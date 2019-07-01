#include <gtest/gtest.h>

#define N 1000

__global__ void vecadd(int a[N], int b[N], int c[N]) {
  int i = threadIdx.x;
  c[i] = a[i] + b[i];
}

int a[N], b[N], c[N], ref[N];

TEST(Cuda, VecAdd) {
  int *cuda_a, *cuda_b, *cuda_c;
  cudaMalloc(&cuda_a, sizeof a);
  cudaMalloc(&cuda_b, sizeof b);
  cudaMalloc(&cuda_c, sizeof c);

  for (int i = 0; i < N; ++i) {
    a[i] = rand() % 100;
    b[i] = rand() % 100;
    ref[i] = a[i] + b[i];
  }

  cudaMemcpy(cuda_a, a, sizeof a, cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_b, b, sizeof b, cudaMemcpyHostToDevice);

  vecadd<<<1, 1024>>>(cuda_a, cuda_b, cuda_c);

  cudaMemcpy(c, cuda_c, sizeof c, cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; ++i) {
    ASSERT_EQ(c[i], ref[i]);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
