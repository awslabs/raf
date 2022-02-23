/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/dispatch/cutlass/operation_table_ext.cu
 * \brief Extentions for cutlass operation table
 */

#include "cutlass_ext/library/operation_table_ext.h"
#include "cutlass_ext/library/conv2d_operation_ext.h"
#include "cutlass_ext/library/gemm_operation_ext.h"

namespace cutlass {
namespace library {

void OperationTableExt::append(Manifest const &manifest) {
  // Insert operations into appropriate data structure
  for (auto const & operation : manifest) {
    OperationDescription const &desc = operation->description();

    // insert all gemm operation into operation table
    if (desc.kind == OperationKind::kGemm) {
      GemmDescriptionExt const &gemm_desc = static_cast<GemmDescriptionExt const &>(desc);
      GemmFunctionalKeyExt functional_key(
        gemm_desc.provider,
        gemm_desc.gemm_kind,
        gemm_desc.tile_description.math_instruction.element_accumulator,
        gemm_desc.element_epilogue,
        gemm_desc.A.element,
        gemm_desc.A.layout,
        gemm_desc.transform_A,
        gemm_desc.B.element,
        gemm_desc.B.layout,
        gemm_desc.transform_B,
        gemm_desc.C.element,
        gemm_desc.epilogue_math_op
      );
      Operation const *op = operation.get();
      int cc = gemm_desc.tile_description.minimum_compute_capability;
      int alignment = std::max(std::max(
        gemm_desc.A.alignment, gemm_desc.B.alignment), gemm_desc.C.alignment);
      GemmPreferenceKey preference_key(cc, alignment);
      gemm_operations[functional_key][preference_key].push_back(op);
    }

    // insert all conv2d operation into operation table
    if (desc.kind == OperationKind::kConv2d) {
      auto &conv_desc = static_cast<ConvDescriptionExt const &>(desc);
      ConvFunctionalKeyExt functional_key(
        conv_desc.provider,
        conv_desc.conv_kind,
        conv_desc.A.element,
        conv_desc.A.layout,
        conv_desc.B.element,
        conv_desc.B.layout,
        conv_desc.C.element,
        conv_desc.C.layout,
        conv_desc.tile_description.math_instruction.element_accumulator,
        conv_desc.element_epilogue,
        conv_desc.epilogue_math_op
      );
      Operation const *op = operation.get();
      int cc = conv_desc.tile_description.minimum_compute_capability;
      ConvPreferenceKey preference_key(cc, conv_desc.iterator_algorithm);
      conv2d_operations[functional_key][preference_key].push_back(op);
    }
  }
}

} // namespace library
} // namespace cutlass
