#include "./list_args.h"
#include "./nn.h"
#include "./ufunc.h"

namespace mnm {
namespace op {
namespace args {
namespace {
MNM_REGISTER_NODE_TYPE(ListArgs);
MNM_REGISTER_NODE_TYPE(UnaryUfuncArgs);
// MNM_REGISTER_NODE_TYPE(UnaryUfuncDxArgs);
MNM_REGISTER_NODE_TYPE(BinaryUfuncArgs);
// MNM_REGISTER_NODE_TYPE(BinaryUfuncDxArgs);
MNM_REGISTER_NODE_TYPE(TernaryUfuncArgs);
// MNM_REGISTER_NODE_TYPE(TernaryUfuncDxArgs);
MNM_REGISTER_NODE_TYPE(UnaryArgs);
MNM_REGISTER_NODE_TYPE(UnaryDxArgs);
MNM_REGISTER_NODE_TYPE(BinaryArgs);
// MNM_REGISTER_NODE_TYPE(BinaryDxArgs);
MNM_REGISTER_NODE_TYPE(TernaryArgs);
// MNM_REGISTER_NODE_TYPE(TernaryArgs);
MNM_REGISTER_NODE_TYPE(ConvArgs);
MNM_REGISTER_NODE_TYPE(ConvDxwArgs);
MNM_REGISTER_NODE_TYPE(PoolArgs);
MNM_REGISTER_NODE_TYPE(PoolDxArgs);
MNM_REGISTER_NODE_TYPE(SoftmaxArgs);
MNM_REGISTER_NODE_TYPE(SoftmaxDxArgs);
MNM_REGISTER_NODE_TYPE(BatchNormArgs);
// MNM_REGISTER_NODE_TYPE(BatchNormDxArgs);
MNM_REGISTER_NODE_TYPE(LocalResponseNormArgs);
// MNM_REGISTER_NODE_TYPE(LocalResponseNormDxArgs);
}  // namespace

std::vector<int64_t> NormalizeTupleOrInt(const value::Value& value) {
  using value::IntValueNode;
  using value::TupleValueNode;
  if (const auto* v = value.as<IntValueNode>()) {
    return {v->data};
  }
  if (const auto* v = value.as<TupleValueNode>()) {
    int n = v->fields.size();
    std::vector<int64_t> result;
    for (int i = 0; i < n; ++i) {
      if (const auto* vv = value.as<IntValueNode>()) {
        result.push_back(vv->data);
      }
      LOG(FATAL) << "Requires tuple of integers, but element " << i << " is not an integer";
      throw;
    }
    return result;
  }
  LOG(FATAL) << "Requires tuple of integers, but get " << value->type_key();
  throw;
}
}  // namespace args
}  // namespace op
}  // namespace mnm
