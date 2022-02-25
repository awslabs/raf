/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/dialect/cutlass/conv_utils.cc
 * \brief Helper functions for cutlass conv
 */
#include "./cutlass_utils.h"
#include "./conv_utils.h"

namespace raf {
namespace op {
namespace cutlass {

void CutlassConvOpEnv::InitConvOperation(
    SplitKMode mode, int N, int H, int W, int C, int K, int R, int S, int pad_h, int pad_w,
    int stride_h, int stride_w, int dilation_h, int dilation_w, NumericTypeID element_accumulator,
    NumericTypeID element_compute, void const* alpha, NumericTypeID element_A,
    LayoutTypeID layout_A, void const* ptr_A, NumericTypeID element_B, LayoutTypeID layout_B,
    void const* ptr_B, void const* beta, NumericTypeID element_C, void const* ptr_C, void* ptr_D,
    EpilogueKindExt epilogue_math_op, const std::string& preferred_name) {
  int P = (H + 2 * pad_h - ((R - 1) * dilation_h + 1)) / stride_h + 1;

  int Q = (W + 2 * pad_w - ((S - 1) * dilation_w + 1)) / stride_w + 1;

  functional_key_ = std::make_unique<ConvFunctionalKeyExt>(
      provider_, ConvKind::kFprop, element_A, layout_A, element_B, layout_B, element_C, layout_A,
      element_accumulator, element_compute, epilogue_math_op);

  auto operators_it = SingletonExt::get().operation_table.conv2d_operations.find(*functional_key_);

  CHECK(operators_it != SingletonExt::get().operation_table.conv2d_operations.end());
  CHECK(!operators_it->second.empty());

  preference_key_ =
      std::make_unique<ConvPreferenceKey>(compute_capability(), IteratorAlgorithmID::kOptimized);

  Operation const* operation =
      find_conv2d_operation(operators_it, *preference_key_, preferred_name);
  CHECK(operation);

  operation_ = operation;

  // Configure operation
  conv::Conv2dProblemSize problem_size =
      conv::Conv2dProblemSize(N, H, W, C, K, R, S, P, Q, pad_h, pad_w, stride_h, stride_w,
                              dilation_h, dilation_w, conv::Mode::kCrossCorrelation);
  // NHWC
  std::vector<int> stride_a = {C, C * W, C * W * H};
  // KRSC
  std::vector<int> stride_b = {C, C * S, C * S * R};
  // NPQK
  std::vector<int> stride_c = {K, K * Q, K * Q * P};
  Conv2dConfiguration configuration{(conv::SplitKMode)mode, problem_size, stride_a, stride_b,
                                    stride_c};

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

  arguments_ = ConvArguments{
      nullptr, nullptr, nullptr, nullptr, alpha, beta, scalar_pointer_mode_,
  };
}

std::vector<std::unique_ptr<TunableConfig>> CutlassConvOpEnv::ListTunableConfigs() {
  // Tunable configuration: kernel_name
  std::vector<std::string> kernel_names;
  auto operators_it = SingletonExt::get().operation_table.conv2d_operations.find(*functional_key_);
  CHECK(operators_it != SingletonExt::get().operation_table.conv2d_operations.end());
  auto cc_it = operators_it->second.upper_bound(*preference_key_);
  while (cc_it != operators_it->second.begin()) {
    --cc_it;
    for (auto const* op : cc_it->second) {
      ConvDescription const& desc = static_cast<ConvDescription const&>(op->description());
      int min_cc = desc.tile_description.minimum_compute_capability;
      int max_cc = desc.tile_description.maximum_compute_capability;
      IteratorAlgorithmID iterator_algorithm = desc.iterator_algorithm;
      if ((min_cc <= preference_key_->compute_capability) &&
          (preference_key_->compute_capability <= max_cc) &&
          (iterator_algorithm <= preference_key_->iterator_algorithm)) {
        kernel_names.push_back(desc.name);
      }
    }
  }
  std::vector<std::unique_ptr<TunableConfig>> rets;
  for (const auto& name : kernel_names) {
    rets.push_back(std::make_unique<ConvTunableConfig>(name));
  }
  return rets;
}

void CutlassConvOpEnv::SetTunableConfig(const std::unique_ptr<TunableConfig>& tunable) {
  tunable_ = *static_cast<ConvTunableConfig*>(tunable.get());
}

const Operation* find_conv2d_operation(ConvOperationFunctionalMapExt::const_iterator operators_it,
                                       ConvPreferenceKey const preference_key,
                                       const std::string& preferred_name) {
  // It selects kernels based on the compute capability and iterator algorithm.
  // The runtime hardware restricts upper bound of the compute capability.
  // The preference key restricts upper bound of the iterator algorithm.
  // Under these two restrictions, operators that make use of the highest compute capability
  // and the best iterator algorithm are selected.
  //
  // For operators with the same compute capability and iterator algorithm,
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

  // Search in descending order of compute capability
  do {
    --cc_it;

    // Search tile sizes in order, for now.
    for (auto const* op : cc_it->second) {
      ConvDescription const& desc = static_cast<ConvDescription const&>(op->description());

      int min_cc = desc.tile_description.minimum_compute_capability;
      int max_cc = desc.tile_description.maximum_compute_capability;

      IteratorAlgorithmID iterator_algorithm = desc.iterator_algorithm;

      if ((min_cc <= preference_key.compute_capability) &&
          (preference_key.compute_capability <= max_cc) &&
          (iterator_algorithm <= preference_key.iterator_algorithm)) {
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
