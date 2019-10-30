#include "./list_args.h"
#include "./nn.h"
#include "./ufunc.h"
namespace mnm {
namespace op {
namespace args {
namespace {
MNM_REGISTER_NODE_TYPE(ListArgs);
MNM_REGISTER_NODE_TYPE(BatchNormArgs);
MNM_REGISTER_NODE_TYPE(ConvArgs);
MNM_REGISTER_NODE_TYPE(ConvDxwArgs);
MNM_REGISTER_NODE_TYPE(LocalResponseNormArgs);
MNM_REGISTER_NODE_TYPE(PoolArgs);
MNM_REGISTER_NODE_TYPE(PoolDxArgs);
MNM_REGISTER_NODE_TYPE(SoftmaxArgs);
MNM_REGISTER_NODE_TYPE(SoftmaxDxArgs);
MNM_REGISTER_NODE_TYPE(BinaryArgs);
MNM_REGISTER_NODE_TYPE(BinaryDxArgs);
MNM_REGISTER_NODE_TYPE(BinaryUfuncArgs);
MNM_REGISTER_NODE_TYPE(TernaryArgs);
MNM_REGISTER_NODE_TYPE(TernaryDxArgs);
MNM_REGISTER_NODE_TYPE(TernaryUfuncArgs);
MNM_REGISTER_NODE_TYPE(UnaryArgs);
MNM_REGISTER_NODE_TYPE(UnaryDxArgs);
MNM_REGISTER_NODE_TYPE(UnaryUfuncArgs);
}
}  // namespace args
}  // namespace op
}  // namespace mnm
