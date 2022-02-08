/*!
 * Copyright (c) 2021 by Contributors
 * \file ./src/op/tvm/tvm_attrs.cc
 * \brief Attributes defined in TVM
 */
#include "./tvm_attrs.h"

namespace mnm {
namespace op {
namespace tvm_dialect {

// unary attrs
TVM_REGISTER_NODE_TYPE(UnaryDxAttr);

// reduce attrs
TVM_REGISTER_NODE_TYPE(SumAttrs);
TVM_REGISTER_NODE_TYPE(MeanDxAttrs);

// transform attrs
TVM_REGISTER_NODE_TYPE(StridedSliceDxAttrs);
TVM_REGISTER_NODE_TYPE(DimAttrs);
TVM_REGISTER_NODE_TYPE(FullAttrs);
TVM_REGISTER_NODE_TYPE(StridedSliceDxAttrs);
TVM_REGISTER_NODE_TYPE(SwapAxisAttrs);

// nn attrs
TVM_REGISTER_NODE_TYPE(Conv2dDxwAttrs);
TVM_REGISTER_NODE_TYPE(Conv2dTransposeDxwAttrs);
TVM_REGISTER_NODE_TYPE(LayerNormAttrs);
TVM_REGISTER_NODE_TYPE(BatchNormAttrs);
TVM_REGISTER_NODE_TYPE(PadAttrs);
TVM_REGISTER_NODE_TYPE(ThresholdAttrs);
TVM_REGISTER_NODE_TYPE(ThresholdDxAttrs);

// optimizer attrs
TVM_REGISTER_NODE_TYPE(SgdAttrs);

}  // namespace tvm_dialect
}  // namespace op
}  // namespace mnm
