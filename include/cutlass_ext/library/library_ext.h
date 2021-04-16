/*!
 * Copyright (c) 2021 by Contributors
 * \file include/cutlass_ext/library/library_ext.h
 * \brief Extentions for cutlass library
 */
#pragma once

#include "cutlass/library/library.h"

namespace cutlass {
namespace library {

/*! \brief Extention for ConvDescription, with epilogue operators information */
struct ConvDescriptionExt : public ConvDescription {
  ConvDescriptionExt(const ConvDescription& op, const EpilogueKind& epilogue_math_op)
      : ConvDescription(op), epilogue_math_op(epilogue_math_op) {
  }

  /*! \brief Epilogue operator information */
  EpilogueKind epilogue_math_op;
};

/*! \brief Extention for GemmDescription, with epilogue operators information */
struct GemmDescriptionExt : public GemmDescription {
  GemmDescriptionExt(const GemmDescription& op, const EpilogueKind& epilogue_math_op)
      : GemmDescription(op), epilogue_math_op(epilogue_math_op) {
  }

  /*! \brief Epilogue operator information */
  EpilogueKind epilogue_math_op;
};

}  // namespace library
}  // namespace cutlass
