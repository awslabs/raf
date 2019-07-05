#include <gtest/gtest.h>

#include <mnm/base.h>
#include <mnm/stream_pool.h>

using mnm::Context;
using mnm::DevType;
using mnm::TemplateToken;
using mnm::stream_pool::Stream;

TEST(CUDA, Basic) {
  Context ctx{DevType::kCUDA(), 0};
  std::shared_ptr<Stream> s_3 = Stream::Get(ctx, 3);
  std::shared_ptr<Stream> s_33 = Stream::Get(ctx, 3);
  std::shared_ptr<Stream> s_0 = Stream::Get(ctx, 0);
  ASSERT_NE(s_3, nullptr);
  ASSERT_NE(s_0, nullptr);
  ASSERT_NE(s_0, s_3);
  ASSERT_EQ(s_3, s_33);
}

static constexpr TemplateToken cudnn = "cudnn";
static constexpr TemplateToken cublas = "cublas";

TEST(CUDA, Name) {
  Context ctx{DevType::kCUDA(), 0};
  std::shared_ptr<Stream> s_cudnn = Stream::Get<cudnn>(ctx, 3);
  std::shared_ptr<Stream> s_cudnn_3 = Stream::Get<cudnn>(ctx, 3);
  std::shared_ptr<Stream> s_cudnn_0 = Stream::Get<cudnn>(ctx, 0);
  std::shared_ptr<Stream> s_cublas = Stream::Get<cublas>(ctx, 3);
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
