/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file include/cutlass_ext/library/operation_table_ext.h
 * \brief Extentions for cutlass operation table
 */
#pragma once

#include <sstream>
#include "cutlass/library/library.h"
#include "cutlass/library/operation_table.h"

#include "./library_ext.h"

namespace cutlass {
namespace library {

/*! \brief Extentions for GemmFunctionalKey, with epilogue information */
struct GemmFunctionalKeyExt : public GemmFunctionalKey {
  /*! \brief Epilogue operator */
  EpilogueKindExt epilogue_math_op;

  GemmFunctionalKeyExt(Provider provider, GemmKind gemm_kind = GemmKind::kGemm,
                       NumericTypeID element_compute = NumericTypeID::kF32,
                       NumericTypeID element_scalar = NumericTypeID::kF32,
                       NumericTypeID element_A = NumericTypeID::kF16,
                       LayoutTypeID layout_A = LayoutTypeID::kColumnMajor,
                       ComplexTransform transform_A = ComplexTransform::kNone,
                       NumericTypeID element_B = NumericTypeID::kF16,
                       LayoutTypeID layout_B = LayoutTypeID::kColumnMajor,
                       ComplexTransform transform_B = ComplexTransform::kNone,
                       NumericTypeID element_C = NumericTypeID::kF16,
                       EpilogueKindExt epilogue_math_op = EpilogueKindExt::kLinearCombination)
      : GemmFunctionalKey(provider, gemm_kind, element_compute, element_scalar, element_A, layout_A,
                          transform_A, element_B, layout_B, transform_B, element_C),
        epilogue_math_op(epilogue_math_op) {
  }

  bool operator==(GemmFunctionalKeyExt const& rhs) const {
    return GemmFunctionalKey::operator==(rhs) && epilogue_math_op == rhs.epilogue_math_op;
  }

  bool operator!=(GemmFunctionalKeyExt const& rhs) const {
    return !(*this == rhs);
  }
};

inline std::ostream& operator<<(std::ostream& out, GemmFunctionalKeyExt const& k) {
  std::stringstream ss;
  ss << "{\n"
     << "         provider: " << to_string(k.provider) << "\n"
     << "        gemm_kind: " << to_string(k.gemm_kind) << "\n"
     << "  element_compute: " << to_string(k.element_compute) << "\n"
     << "   element_scalar: " << to_string(k.element_scalar) << "\n"
     << "        element_A: " << to_string(k.element_A) << "\n"
     << "         layout_A: " << to_string(k.layout_A) << "\n"
     << "      transform_A: " << to_string(k.transform_A) << "\n"
     << "        element_B: " << to_string(k.element_B) << "\n"
     << "         layout_B: " << to_string(k.layout_B) << "\n"
     << "      transform_B: " << to_string(k.transform_B) << "\n"
     << "        element_C: " << to_string(k.element_C) << "\n"
     << " epilogue_math_op: " << k.epilogue_math_op << "\n"
     << "}";

  out << ss.str();
  return out;
}

struct GemmFunctionalKeyHasherExt : public GemmFunctionalKeyHasher {
  using IntHash = std::hash<int>;

  inline size_t operator()(GemmFunctionalKeyExt const& key) const {
    IntHash hash;
    return GemmFunctionalKeyHasher::operator()(key) ^ rotl(hash(int(key.epilogue_math_op)), 12);
  }
};

/*! \brief Maps a GemmFunctionalKeyExt onto a vector of Operation* */
using GemmOperationFunctionalMapExt =
    std::unordered_map<GemmFunctionalKeyExt, GemmOperationVectorMap, GemmFunctionalKeyHasherExt>;

/*! \brief Extentions for ConvFunctionalKey, with epilogue information */
struct ConvFunctionalKeyExt : public ConvFunctionalKey {
  /*! \brief Epilogue operator */
  EpilogueKindExt epilogue_math_op;

  ConvFunctionalKeyExt(library::Provider provider = library::Provider::kInvalid,
                       library::ConvKind conv_kind = library::ConvKind::kFprop,
                       library::NumericTypeID element_A = library::NumericTypeID::kF16,
                       library::LayoutTypeID layout_A = library::LayoutTypeID::kTensorNHWC,
                       library::NumericTypeID element_B = library::NumericTypeID::kF16,
                       library::LayoutTypeID layout_B = library::LayoutTypeID::kTensorNHWC,
                       library::NumericTypeID element_C = library::NumericTypeID::kF16,
                       library::LayoutTypeID layout_C = library::LayoutTypeID::kTensorNHWC,
                       library::NumericTypeID element_accumulator = library::NumericTypeID::kF32,
                       library::NumericTypeID element_compute = library::NumericTypeID::kF32,
                       EpilogueKindExt epilogue_math_op = EpilogueKindExt::kLinearCombination)
      : ConvFunctionalKey(provider, conv_kind, element_A, layout_A, element_B, layout_B, element_C,
                          layout_C, element_accumulator, element_compute),
        epilogue_math_op(epilogue_math_op) {
  }

  bool operator==(ConvFunctionalKeyExt const& rhs) const {
    return ConvFunctionalKey::operator==(rhs) && epilogue_math_op == rhs.epilogue_math_op;
  }

  bool operator!=(ConvFunctionalKeyExt const& rhs) const {
    return !(*this == rhs);
  }
};

struct ConvFunctionalKeyHasherExt : ConvFunctionalKeyHasher {
  using IntHash = std::hash<int>;

  size_t operator()(ConvFunctionalKeyExt const& key) const {
    IntHash hash;
    return ConvFunctionalKeyHasher::operator()(key) ^ rotl(hash(int(key.epilogue_math_op)), 11);
  }
};

/*! \brief Maps a ConvFunctionalKeyExt onto a vector of Operation* */
using ConvOperationFunctionalMapExt =
    std::unordered_map<ConvFunctionalKeyExt, ConvOperationVectorMap, ConvFunctionalKeyHasherExt>;

/*!
 * \brief Table of cutlass::library::Operation instances,
 *        including operators with epilogue
 */
class OperationTableExt {
 public:
  /*! \brief Map of all operations of type kGemm */
  GemmOperationFunctionalMapExt gemm_operations;

  /*! \brief Map of all operations of type kConv2d */
  ConvOperationFunctionalMapExt conv2d_operations;

 public:
  /*! \brief Append all operations in manifest to the operation table */
  void append(Manifest const& manifest);
};

}  // namespace library
}  // namespace cutlass
