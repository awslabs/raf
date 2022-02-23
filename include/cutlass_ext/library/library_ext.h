/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file include/cutlass_ext/library/library_ext.h
 * \brief Extentions for cutlass library
 */
#pragma once

#include "cutlass/library/library.h"

namespace cutlass {
namespace library {

enum class EpilogueKindExt {
  kUnknown,
  kConversion,
  kLinearCombination,
  kLinearCombinationClamp,
  kLinearCombinationPlanarComplex,
  kLinearCombinationRelu,
  kLinearCombinationSigmoid,
  kLinearCombinationGelu,
  kInvalid
};

inline std::ostream& operator<<(std::ostream& out, const EpilogueKindExt& value) {
  std::string str;
#define PROCESS_VAL(p) \
  case (p):            \
    str = "#p";        \
    break;

  switch (value) {
    PROCESS_VAL(EpilogueKindExt::kUnknown);
    PROCESS_VAL(EpilogueKindExt::kConversion);
    PROCESS_VAL(EpilogueKindExt::kLinearCombination);
    PROCESS_VAL(EpilogueKindExt::kLinearCombinationClamp);
    PROCESS_VAL(EpilogueKindExt::kLinearCombinationPlanarComplex);
    PROCESS_VAL(EpilogueKindExt::kLinearCombinationRelu);
    PROCESS_VAL(EpilogueKindExt::kLinearCombinationSigmoid);
    PROCESS_VAL(EpilogueKindExt::kLinearCombinationGelu);
    PROCESS_VAL(EpilogueKindExt::kInvalid);
    default:
      str = "internal error";
  }
#undef PROCESS_VAL

  return out << str;
}

/*! \brief Extention for ConvDescription, with epilogue operators information */
struct ConvDescriptionExt : public ConvDescription {
  ConvDescriptionExt(const ConvDescription& op, const EpilogueKindExt& epilogue_math_op)
      : ConvDescription(op), epilogue_math_op(epilogue_math_op) {
  }

  /*! \brief Epilogue operator information */
  EpilogueKindExt epilogue_math_op;
};

/*! \brief Extention for GemmDescription, with epilogue operators information */
struct GemmDescriptionExt : public GemmDescription {
  GemmDescriptionExt(const GemmDescription& op, const EpilogueKindExt& epilogue_math_op)
      : GemmDescription(op), epilogue_math_op(epilogue_math_op) {
  }

  /*! \brief Epilogue operator information */
  EpilogueKindExt epilogue_math_op;
};

}  // namespace library
}  // namespace cutlass
