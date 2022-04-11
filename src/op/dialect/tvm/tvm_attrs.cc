/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file ./src/op/tvm/tvm_attrs.cc
 * \brief Attributes defined in TVM
 */
#include "./tvm_attrs.h"

namespace raf {
namespace op {
namespace tvm_dialect {

// unary attrs
RAF_REGISTER_OBJECT_REFLECT(UnaryDxAttr);

// reduce attrs
RAF_REGISTER_OBJECT_REFLECT(SumAttrs);
RAF_REGISTER_OBJECT_REFLECT(MeanDxAttrs);

// transform attrs
RAF_REGISTER_OBJECT_REFLECT(DimAttrs);
RAF_REGISTER_OBJECT_REFLECT(FullAttrs);
RAF_REGISTER_OBJECT_REFLECT(StridedSliceDxAttrs);
RAF_REGISTER_OBJECT_REFLECT(SwapAxisAttrs);

// nn attrs
RAF_REGISTER_OBJECT_REFLECT(Conv2dDxwAttrs);
RAF_REGISTER_OBJECT_REFLECT(Conv2dTransposeDxwAttrs);
RAF_REGISTER_OBJECT_REFLECT(LayerNormAttrs);
RAF_REGISTER_OBJECT_REFLECT(BatchNormAttrs);
RAF_REGISTER_OBJECT_REFLECT(PadAttrs);
RAF_REGISTER_OBJECT_REFLECT(ThresholdAttrs);
RAF_REGISTER_OBJECT_REFLECT(ThresholdDxAttrs);

// optimizer attrs
RAF_REGISTER_OBJECT_REFLECT(SgdAttrs);

}  // namespace tvm_dialect
}  // namespace op
}  // namespace raf
