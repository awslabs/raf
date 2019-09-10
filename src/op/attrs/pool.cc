#include <mnm/ir.h>

#include "./pool.h"

namespace mnm {
namespace op {
namespace attrs {

MNM_REGISTER_NODE_TYPE(MaxPoolAttrs);
MNM_REGISTER_NODE_TYPE(AvgPoolAttrs);
MNM_REGISTER_NODE_TYPE(MaxPoolBackAttrs);
MNM_REGISTER_NODE_TYPE(AvgPoolBackAttrs);

}  // namespace attrs
}  // namespace op
}  // namespace mnm
