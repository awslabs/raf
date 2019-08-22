#pragma once

#include <mnm/memory_pool.h>
#include <mnm/op.h>
#include <mnm/stream_pool.h>
#include <mnm/value.h>

namespace mnm {
namespace requests {
class Requests;
}  // namespace requests
}  // namespace mnm

namespace mnm {
namespace executor {

class Executor {
 public:
  virtual ~Executor() = default;
  virtual void OnBind(const op::OpEnv* op_env) = 0;
  virtual void OnDestruct(const op::OpEnv* op_env) = 0;
  virtual void OnBind(const value::BoundExprNode* bound_expr) = 0;
  virtual void OnDestruct(const value::BoundExprNode* bound_expr) = 0;
  virtual void RequestMemory(requests::Requests* request, int index) = 0;
  virtual void RequestWorkspace(requests::Requests* request, int index) = 0;
  virtual void RequestStream(requests::Requests* request, int index) = 0;
  virtual void RequestDistributed(requests::Requests* request, int index) {
    LOG(FATAL) << "NotImplementedError: RequestDistributed";
    throw;
  }
};

}  // namespace executor
}  // namespace mnm
