/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
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
