/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/dialect/cutlass/conv_utils.h
 * \brief Helper functions for cutlass conv
 */
#pragma once

#include "./cutlass_utils.h"

namespace raf {
namespace op {
namespace cutlass {

/*! \brief Tunable configurations for cutlass conv */
struct ConvTunableConfig : public TunableConfig {
  ConvTunableConfig(std::string kernel_name) : TunableConfig(kernel_name) {
  }

  ConvTunableConfig() {
  }

  virtual void AsText(std::ostream& os) const override {
    os << "{" << std::endl;
    os << "  kernel_name: " << kernel_name << std::endl;
    os << "}" << std::endl;
  }
};

class CutlassConvOpEnv : public CutlassOpEnv {
 public:
  explicit CutlassConvOpEnv(const CallValues& call) : CutlassOpEnv(call) {
  }

  std::vector<std::unique_ptr<TunableConfig>> ListTunableConfigs() override;

  void SetTunableConfig(const std::unique_ptr<TunableConfig>& tunable) override;

  /*!
   * \brief Initialize a convolution operator
   * \param N Batch size
   * \param H Input height
   * \param W Input width
   * \param C Input channel
   * \param K Output channel
   * \param R Filter height
   * \param S Filter width
   * \param pad_h Height padding
   * \param pad_w Width padding
   * \param stride_h Height stride
   * \param stride_w Width stride
   * \param dilation_h Height dilation
   * \param dilation_w Width dilation
   * \param element_accumulator Data type of internal accumulation
   * \param element_compute Data type of alpha/beta scalars
   * \param alpha Pointer to alpha scalar
   * \param element_A Data type of A matrix elements
   * \param layout_A Layout of A matrix
   * \param ptr_A Pointer to A matrix in Global Memory
   * \param element_B Data type of B matrix elements
   * \param layout_B Layout of B matrix
   * \param ptr_B Pointer to B matrix in Global Memory
   * \param beta Pointer to beta scalar
   * \param element_C Data type of C and D matrices
   * \param ptr_C Pointer to C matrix
   * \param ptr_D Pointer to D matrix
   * \param epilogue_math_op Epilogue operator
   * \param preferred_name Preferred kernel name.
                           See the implementation of find_conv2d_operation for details
   */
  void InitConvOperation(SplitKMode mode, int N, int H, int W, int C, int K, int R, int S,
                         int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h,
                         int dilation_w, NumericTypeID element_accumulator,
                         NumericTypeID element_compute, void const* alpha, NumericTypeID element_A,
                         LayoutTypeID layout_A, void const* ptr_A, NumericTypeID element_B,
                         LayoutTypeID layout_B, void const* ptr_B, void const* beta,
                         NumericTypeID element_C, void const* ptr_C, void* ptr_D,
                         EpilogueKindExt epilogue_math_op = EpilogueKindExt::kLinearCombination,
                         const std::string& preferred_name = "");

 protected:
  /*! \brief Convolution operator arguments */
  ConvArguments arguments_;
  /*! \brief Conv functional key */
  std::unique_ptr<ConvFunctionalKeyExt> functional_key_;
  /*! \brief Conv functional key */
  std::unique_ptr<ConvPreferenceKey> preference_key_;
  /*! \brief Tunable configuration for cutlass conv */
  ConvTunableConfig tunable_;
};

/*!
 * \brief Find the best kernel in descending order of preference.
 * \param operators_it An iterator for all valid operators
 * \param preference_key Describes the preferred operator
 * \param preferred_name Describes the name of the preferred operator
 */
Operation const* find_conv2d_operation(ConvOperationFunctionalMapExt::const_iterator operators_it,
                                       ConvPreferenceKey const preference_key,
                                       const std::string& preferred_name = "");

}  // namespace cutlass
}  // namespace op
}  // namespace raf
