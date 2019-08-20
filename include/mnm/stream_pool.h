#pragma once

#include <memory>

#include <mnm/base.h>

namespace mnm {
namespace stream_pool {

class Tag final {
 public:
  Tag(const std::string& data) : data(data) {
    index = GetTagIndex_(data);
  }

 public:
  std::string data;
  int index;

 private:
  static int GetTagIndex_(const std::string& tag);
};

class Stream final {
 public:
  class Impl;
  friend Impl;

 public:
  Stream() = default;

  Stream(Impl* impl);

  ~Stream();

  void* data() const;

  static std::shared_ptr<Stream> Get(const Context& ctx, int tag_idx, int index);

 private:
  std::unique_ptr<Impl> impl;
};

}  // namespace stream_pool
}  // namespace mnm
