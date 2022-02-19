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
