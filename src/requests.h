#pragma once

#include <mnm/base.h>
#include <mnm/memory_pool.h>
#include <mnm/stream_pool.h>

namespace mnm {
namespace requests {

class Requests {
 public:
  struct MemoryRequest {
    void** dest;
    Context ctx;
    int64_t nbytes;
    std::shared_ptr<memory_pool::Memory> memory;
  };

  struct WorkspaceRequest {
    void** dest;
    Context ctx;
    int64_t nbytes;
    std::shared_ptr<memory_pool::Memory> memory;
  };

  struct StreamRequest {
    void** dest;
    Context ctx;
    int tag_idx;
    int stream_idx;
    std::shared_ptr<stream_pool::Stream> stream;
  };

 public:
  std::vector<MemoryRequest> memory;
  std::vector<WorkspaceRequest> workspace;
  std::vector<StreamRequest> stream;
};

}  // namespace requests
}  // namespace mnm
