#pragma once

#include <mnm/base.h>
#include <mnm/value.h>

namespace mnm {
namespace requests {

class Requests {
 public:
  struct MemoryRequest {
    value::Value& value;
    Context ctx;
    int64_t nbytes;
  };

  struct WorkspaceRequest {
    void** dest;
    Context ctx;
    int64_t nbytes;
  };

  struct StreamRequest {
    void** dest;
    Context ctx;
    int tag_idx;
    int index;
  };

 public:
  std::vector<MemoryRequest> memory;
  std::vector<WorkspaceRequest> workspace;
  std::vector<StreamRequest> stream;
};

}  // namespace requests
}  // namespace mnm
