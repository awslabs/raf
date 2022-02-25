/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/dialect/cutlass/gemm_utils.cc
 * \brief Helper functions for cutlass gemm
 */
#include <sstream>
#include "./cutlass_utils.h"
#include "./gemm_utils.h"

namespace raf {
namespace op {
namespace cutlass {

void CutlassGemmOpEnv::InitGemmOperation(
    GemmUniversalMode mode, int M, int N, int K, NumericTypeID element_compute,
    NumericTypeID element_scalar, void const* alpha, NumericTypeID element_A, LayoutTypeID layout_A,
    void const* ptr_A, int lda, NumericTypeID element_B, LayoutTypeID layout_B, void const* ptr_B,
    int ldb, void const* beta, NumericTypeID element_C, void const* ptr_C, int ldc, void* ptr_D,
    int ldd, int batch_count, int64_t batch_stride_A, int64_t batch_stride_B,
    int64_t batch_stride_C, int64_t batch_stride_D, EpilogueKindExt epilogue_math_op,
    const std::string& preferred_name) {
  functional_key_ = std::make_unique<GemmFunctionalKeyExt>(
      provider_, GemmKind::kUniversal, element_compute, element_scalar, element_A, layout_A,
      ComplexTransform::kNone, element_B, layout_B, ComplexTransform::kNone, element_C,
      epilogue_math_op);
  auto operators_it = SingletonExt::get().operation_table.gemm_operations.find(*functional_key_);
  if (operators_it == SingletonExt::get().operation_table.gemm_operations.end()) {
    std::stringstream ss;
    ss << "Cannot find the required GEMM op in CUTLASS with the functional key:\n"
       << *functional_key_ << "\nAvailable keys:\n";
    for (auto pair : SingletonExt::get().operation_table.gemm_operations) {
      ss << "  " << pair.first << "\n";
    }
    LOG(FATAL) << ss.str();
  }

  CHECK(!operators_it->second.empty());

  // Maximum alignment expectation among all kernels (in units of bytes)
  int const kMaximumAlignmentSize = 16;
  void const* ptr_A_check = ptr_A;
  void const* ptr_B_check = ptr_B;
  void const* ptr_C_check = ptr_C;
  void* ptr_D_check = ptr_D;
  int alignment = gemm_problem_alignment(M, N, K, element_A, ptr_A_check, lda, 0, element_B,
                                         ptr_B_check, ldb, 0, element_C, ptr_C_check, ldc, 0,
                                         ptr_D_check, ldd, 0, kMaximumAlignmentSize);

  // Find the best kernel in descending order of preference.
  preference_key_ = std::make_unique<GemmPreferenceKey>(compute_capability(), alignment);
  Operation const* operation = find_gemm_operation(operators_it, *preference_key_, preferred_name);
  CHECK(operation);
  operation_ = operation;

  // Configure operation
  GemmUniversalConfiguration configuration{mode, {M, N, K}, batch_count, lda, ldb, ldc, ldd};

  // Query host work space size
  uint64_t host_workspace_size_needed = operation->get_host_workspace_size(&configuration);
  CHECK_GE(uint64_t(kHostWorkspaceSize), host_workspace_size_needed);

  // Query device workspace size
  workspace_size_ = operation->get_device_workspace_size(&configuration);
  if (workspace_size_ > 0) {
    RequestWorkspace(&workspace_, device_, workspace_size_);
  }

  // Initialize host and device workspaces
  CUTLASS_CALL(operation->initialize(&configuration, host_workspace_, workspace_, GetStream()));
  arguments_ = GemmUniversalArguments{nullptr,
                                      nullptr,
                                      nullptr,
                                      nullptr,
                                      alpha,
                                      beta,
                                      scalar_pointer_mode_,
                                      batch_stride_A,
                                      batch_stride_B,
                                      batch_stride_C,
                                      batch_stride_D};
}

std::vector<std::unique_ptr<TunableConfig>> CutlassGemmOpEnv::ListTunableConfigs() {
  // Tunable configuration: split_k_slices
  // Split axis k into 1 slice (no slicing) or 4 slices
  const static std::vector<int> split_k_slices = {1, 2, 4, 8};

  // Tunable configuration: split_k_mode
  // Whether to compute different slices serially or parallelly
  const static std::vector<SplitKMode> split_k_mode = {SplitKMode::kSerial};

  // Tunable configuration: kernel_name
  std::vector<std::string> kernel_names;
  auto operators_it = SingletonExt::get().operation_table.gemm_operations.find(*functional_key_);
  CHECK(operators_it != SingletonExt::get().operation_table.gemm_operations.end());
  auto cc_it = operators_it->second.upper_bound(*preference_key_);
  while (cc_it != operators_it->second.begin()) {
    --cc_it;
    for (auto const* op : cc_it->second) {
      GemmDescription const& desc = static_cast<GemmDescription const&>(op->description());
      int min_cc = desc.tile_description.minimum_compute_capability;
      int max_cc = desc.tile_description.maximum_compute_capability;
      int op_alignment = maximum_alignment_requirement(desc);
      if ((min_cc <= preference_key_->compute_capability) &&
          (preference_key_->compute_capability <= max_cc) &&
          (op_alignment <= preference_key_->alignment)) {
        kernel_names.push_back(desc.name);
      }
    }
  }
  std::vector<std::unique_ptr<TunableConfig>> rets;
  for (const auto& name : kernel_names) {
    for (const auto& i_split_k_slices : split_k_slices) {
      for (const auto& i_split_k_mode : split_k_mode) {
        rets.push_back(std::make_unique<GemmTunableConfig>(name, i_split_k_mode, i_split_k_slices));
      }
    }
  }
  return rets;
}

void CutlassGemmOpEnv::SetTunableConfig(const std::unique_ptr<TunableConfig>& tunable) {
  tunable_ = *static_cast<GemmTunableConfig*>(tunable.get());
}

int gemm_problem_alignment(int M, int N, int K, NumericTypeID element_A, void const* ptr_A, int lda,
                           int64_t batch_stride_A, NumericTypeID element_B, void const* ptr_B,
                           int ldb, int64_t batch_stride_B, NumericTypeID element_C,
                           void const* ptr_C, int ldc, int64_t batch_stride_C, void const* ptr_D,
                           int ldd, int64_t batch_stride_D, int max_alignment_in_bytes) {
  void const* pointers[] = {ptr_A, ptr_B, ptr_C, ptr_D};
  int64_t extents[] = {
      M, N, K, lda, ldb, ldc, ldd, batch_stride_A, batch_stride_B, batch_stride_C, batch_stride_D};
  NumericTypeID elements[] = {element_A, element_B, element_C};
  for (; max_alignment_in_bytes > 0; max_alignment_in_bytes /= 2) {
    bool satisfied = true;
    for (void const* ptr : pointers) {
      std::uintptr_t int_ptr = reinterpret_cast<std::uintptr_t>(ptr);
      if (int_ptr % max_alignment_in_bytes) {
        satisfied = false;
        break;
      }
    }
    if (!satisfied) {
      continue;
    }
    int max_element_alignment = 0;
    for (NumericTypeID type_id : elements) {
      int element_alignment = max_alignment_in_bytes * 8 / library::sizeof_bits(type_id);
      max_element_alignment = std::max(max_element_alignment, element_alignment);
    }
    for (int64_t extent : extents) {
      if (extent % max_element_alignment) {
        satisfied = false;
        break;
      }
    }
    if (!satisfied) {
      continue;
    }
    return max_element_alignment;
  }

  // No alignment satisfies this problem
  return 0;
}

int maximum_alignment_requirement(GemmDescription const& desc) {
  return std::max(std::max(desc.A.alignment, desc.B.alignment), desc.C.alignment);
}

Operation const* find_gemm_operation(GemmOperationFunctionalMapExt::const_iterator operators_it,
                                     GemmPreferenceKey const preference_key,
                                     const std::string& preferred_name) {
  // It selects kernels based on the compute capability and address alignment.
  // The runtime hardware restricts upper bound of the compute capability.
  // The address of the input tensors restricts upper bound of the memory alignment.
  // Under these two restrictions, operators that make use of the highest compute capability
  // and highest alignment are selected.
  //
  // For operators with the same compute capability and memory alignment,
  // an arbitrary one is selected.
  //
  // If preferred_name is designated, the operator with the preferred name is selected.
  auto cc_it = operators_it->second.upper_bound(preference_key);
  if (cc_it == operators_it->second.begin()) {
    // The operator preference key is an rough indicator for its performance:
    // The larger its key, the better its performance.
    // The designated preference_key is the best possible key:
    // keys <= preference_key are valid, keys > preference_key are invalid
    // `cc_it == operators_it->second.begin()` shows that there are no valid keys.
    return nullptr;
  }
  Operation const* operation = nullptr;
  do {
    --cc_it;
    for (auto const* op : cc_it->second) {
      GemmDescription const& desc = static_cast<GemmDescription const&>(op->description());
      int min_cc = desc.tile_description.minimum_compute_capability;
      int max_cc = desc.tile_description.maximum_compute_capability;
      int op_alignment = maximum_alignment_requirement(desc);
      if ((min_cc <= preference_key.compute_capability) &&
          (preference_key.compute_capability <= max_cc) &&
          (op_alignment <= preference_key.alignment)) {
        if (operation == nullptr) {
          operation = op;
        } else if (desc.name == preferred_name) {
          operation = op;
        }
      }
    }
  } while (!operation && cc_it != operators_it->second.begin());
  return operation;
}

}  // namespace cutlass
}  // namespace op
}  // namespace raf
