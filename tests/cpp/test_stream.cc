#include <gtest/gtest.h>

#include <mnm/base.h>
#include <mnm/stream_pool.h>

using mnm::Context;
using mnm::DevType;
using mnm::stream_pool::Stream;
using mnm::stream_pool::Tag;

static const Tag normal("");
static const Tag cudnn("cudnn");
static const Tag cublas("cublas");

TEST(CUDA, Basic) {
  Context ctx{DevType::kCUDA(), 0};
  std::shared_ptr<Stream> s_3 = Stream::Get(ctx, normal.index, 3);
  std::shared_ptr<Stream> s_33 = Stream::Get(ctx, normal.index, 3);
  std::shared_ptr<Stream> s_0 = Stream::Get(ctx, normal.index, 0);
  ASSERT_NE(s_3, nullptr);
  ASSERT_NE(s_0, nullptr);
  ASSERT_NE(s_0, s_3);
  ASSERT_EQ(s_3, s_33);
}

TEST(CUDA, Name) {
  Context ctx{DevType::kCUDA(), 0};
  std::shared_ptr<Stream> s_cudnn = Stream::Get(ctx, cudnn.index, 3);
  std::shared_ptr<Stream> s_cudnn_3 = Stream::Get(ctx, cudnn.index, 3);
  std::shared_ptr<Stream> s_cudnn_0 = Stream::Get(ctx, cudnn.index, 0);
  std::shared_ptr<Stream> s_cublas = Stream::Get(ctx, cublas.index, 3);
  ASSERT_NE(s_cudnn, nullptr);
  ASSERT_NE(s_cublas, nullptr);
  ASSERT_NE(s_cudnn, s_cublas);
  ASSERT_EQ(s_cudnn, s_cudnn_3);
  ASSERT_NE(s_cudnn_0, s_cudnn_3);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
