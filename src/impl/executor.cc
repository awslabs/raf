#include <mnm/executor.h>

#include "../requests.h"

namespace mnm {
namespace executor {
std::unique_ptr<requests::Requests> Executor::AttachOpEnv(op::OpEnv* env) {
  return env->SetExecutor(this);
}
}  // namespace executor
}  // namespace mnm
