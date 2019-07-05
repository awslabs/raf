#pragma once

#include <memory>

#include <mnm/base.h>

namespace mnm {
namespace stream_pool {

class Stream final {
 public:
  class Impl;
  friend Impl;

 public:
  Stream() = default;

  Stream(Impl* impl);

  ~Stream();

  template <const TemplateToken& tag>
  static std::shared_ptr<Stream> Get(const Context& ctx, int index) {
    static int tag_index = GetTagIndex(tag);
    return Get(ctx, index, tag_index);
  }

  static std::shared_ptr<Stream> Get(const Context& ctx, int index) {
    static int tag_index = GetTagIndex("");
    return Get(ctx, index, tag_index);
  }

 private:
  static int GetTagIndex(const std::string& tag);

  static std::shared_ptr<Stream> Get(const Context& ctx, int tag_index, int index);

 private:
  std::unique_ptr<Impl> impl;
};

}  // namespace stream_pool
}  // namespace mnm
