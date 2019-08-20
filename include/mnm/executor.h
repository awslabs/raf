#pragma once

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
  std::unique_ptr<requests::Requests> AttachOpEnv(op::OpEnv* env);
  virtual ~Executor() = default;
  virtual void OnBind(const value::BoundExprNode* bound_expr) = 0;
  virtual void OnDestruct(const value::ValueNode* value) = 0;
  virtual void OnDestruct(const value::BoundExprNode* bound_expr) = 0;
  virtual void RequestMemory(value::Value& value, const Context& ctx, int64_t nbytes) = 0;
  virtual void RequestWorkspace(void** dest, const Context& ctx, int64_t nbytes) = 0;
  virtual void RequestStream(void** dest, const Context& ctx, int tag_idx, int index) = 0;
  virtual void RequestDistributed(void** dest) {
    LOG(FATAL) << "NotImplementedError: RequestDistributed";
    throw;
  }
};

}  // namespace executor
}  // namespace mnm
