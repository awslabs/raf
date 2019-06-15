#pragma once

#include <mnm/types.h>

namespace mnm {
namespace requests {

class Requests {
 public:
  struct MemoryRequest {
    void** dest;
    mnm::types::Context ctx;
    int64_t nbytes;
  };

  struct WorkspaceRequest {
    void** dest;
    mnm::types::Context ctx;
    int64_t nbytes;
  };

  struct StreamRequest {
    void** dest;
    mnm::types::Context ctx;
  };

  struct DistRequest {
    void** dest;
  };

 public:
  std::vector<MemoryRequest> memory;
  std::vector<WorkspaceRequest> workspace;
  std::vector<StreamRequest> stream;
  std::vector<DistRequest> dist;
};

}  // namespace requests
}  // namespace mnm
