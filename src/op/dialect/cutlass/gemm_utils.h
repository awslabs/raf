/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/dialect/cutlass/gemm_utils.h
 * \brief Helper functions for cutlass gemm
 */
#pragma once

#include "./cutlass_utils.h"

namespace raf {
namespace op {
namespace cutlass {

/*! \brief Tunable configurations for cutlass gemm */
struct GemmTunableConfig : public TunableConfig {
  GemmTunableConfig(std::string kernel_name, SplitKMode split_k_mode, int split_k_slices)
      : TunableConfig(kernel_name), split_k_mode(split_k_mode), split_k_slices(split_k_slices) {
  }

  GemmTunableConfig() : split_k_mode(SplitKMode::kSerial), split_k_slices(1) {
  }

  virtual void AsText(std::ostream& os) const override {
    os << "{" << std::endl;
    os << "  kernel_name: " << kernel_name << std::endl;
    os << "  split_k_mode: " << split_k_mode << std::endl;
    os << "  split_k_slices: " << split_k_slices << std::endl;
    os << "}" << std::endl;
  }

  /*! \brief split axis k serially or parallelly */
  SplitKMode split_k_mode;
  /*! \brief the number of slices to be split into */
  int split_k_slices;
};

class CutlassGemmOpEnv : public CutlassOpEnv {
 public:
  explicit CutlassGemmOpEnv(const CallValues& call) : CutlassOpEnv(call) {
  }

  std::vector<std::unique_ptr<TunableConfig>> ListTunableConfigs() override;

  void SetTunableConfig(const std::unique_ptr<TunableConfig>& tunable) override;

  /*!
   * \brief Initialize a gemm operator
   * \param mode GEMM mode
   * \param M GEMM M dimension
   * \param N GEMM N dimension
   * \param K GEMM K dimension
   * \param element_compute Data type of internal accumulation
   * \param element_scalar  Data type of alpha/beta scalars
   * \param alpha Pointer to alpha scalar
   * \param element_A Data type of A matrix elements
   * \param layout_A Layout of A matrix
   * \param ptr_A Pointer to A matrix in Global Memory
   * \param lda Leading dimension of A matrix
   * \param element_B Data type of B matrix elements
   * \param layout_B Layout of B matrix
   * \param ptr_B Pointer to B matrix in Global Memory
   * \param ldb Leading dimension of B matrix
   * \param beta Pointer to beta scalar
   * \param element_C Data type of C and D matrices
   * \param ptr_C Pointer to C matrix
   * \param ldc Leading dimension of C matrix
   * \param ptr_D Pointer to D matrix
   * \param ldd Leading dimension of D matrix
   * \param batch_count Batch count or number of split-K slices
   * \param batch_stride_A Batch stride of A operand
   * \param batch_stride_B Batch stride of B operand
   * \param batch_stride_C Batch stride of C operand
   * \param batch_stride_D Batch stride of D operand
   * \param epilogue_math_op Epilogue operator
   * \param preferred_name Preferred kernel name.
   *                       See the implementation of find_gemm_operation for details
   */
  void InitGemmOperation(GemmUniversalMode mode, int M, int N, int K, NumericTypeID element_compute,
                         NumericTypeID element_scalar, void const* alpha, NumericTypeID element_A,
                         LayoutTypeID layout_A, void const* ptr_A, int lda, NumericTypeID element_B,
                         LayoutTypeID layout_B, void const* ptr_B, int ldb, void const* beta,
                         NumericTypeID element_C, void const* ptr_C, int ldc, void* ptr_D, int ldd,
                         int batch_count = 1, int64_t batch_stride_A = 0,
                         int64_t batch_stride_B = 0, int64_t batch_stride_C = 0,
                         int64_t batch_stride_D = 0,
                         EpilogueKindExt epilogue_math_op = EpilogueKindExt::kLinearCombination,
                         const std::string& preferred_name = "");

 protected:
  /*! \brief Gemm operator arguments */
  GemmUniversalArguments arguments_;
  /*! \brief Gemm functional key */
  std::unique_ptr<GemmFunctionalKeyExt> functional_key_;
  /*! \brief Gemm functional key */
  std::unique_ptr<GemmPreferenceKey> preference_key_;
  /*! \brief Tunable configuration for cutlass gemm */
  GemmTunableConfig tunable_;
};

/*!
 * \brief Returns the largest alignment (in units of elements) the problem satisfies,
 * starting from a given upper limit.
 * \param M GEMM M dimension
 * \param N GEMM N dimension
 * \param K GEMM K dimension
 * \param element_A Data type of A matrix elements
 * \param ptr_A Pointer to A matrix in Global Memory
 * \param lda Leading dimension of A matrix
 * \param batch_stride_A Batch stride of A operand
 * \param element_B Data type of B matrix elements
 * \param ptr_B Pointer to B matrix in Global Memory
 * \param ldb Leading dimension of B matrix
 * \param batch_stride_B Batch stride of B operand
 * \param element_C Data type of C and D matrices
 * \param ptr_C Pointer to C matrix
 * \param ldc Leading dimension of C matrix
 * \param batch_stride_C Batch stride of C operand
 * \param ptr_D Pointer to D matrix
 * \param ldd Leading dimension of D matrix
 * \param batch_stride_D Batch stride of D operand
 * \param max_alignment_in_bytes the given alignment upper limit
 * \return the largest alignment (in units of elements) the problem satisfies
 */
int gemm_problem_alignment(int M, int N, int K, NumericTypeID element_A, void const* ptr_A, int lda,
                           int64_t batch_stride_A, NumericTypeID element_B, void const* ptr_B,
                           int ldb, int64_t batch_stride_B, NumericTypeID element_C,
                           void const* ptr_C, int ldc, int64_t batch_stride_C, void const* ptr_D,
                           int ldd, int64_t batch_stride_D, int max_alignment_in_bytes = 16);

/*!
 * \brief Returns the maximum required alignment for each operator
 * \param desc Gemm operation descriptor
 * \return the maximum required alignment for each operator
 */
int maximum_alignment_requirement(GemmDescription const& desc);

/*!
 * \brief Find the best kernel in descending order of preference.
 * \param operators_it An iterator for all valid operators
 * \param preference_key Describes the preferred operator
 * \param preferred_name Describes the name of the preferred operator
 */
const Operation* find_gemm_operation(GemmOperationFunctionalMapExt::const_iterator operators_it,
                                     GemmPreferenceKey const preference_key,
                                     const std::string& preferred_name = "");

}  // namespace cutlass
}  // namespace op
}  // namespace raf
