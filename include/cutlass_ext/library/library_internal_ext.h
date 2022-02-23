/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file include/cutlass_ext/library/library_internal_ext.h
 * \brief Extentions for cutlass library internal classes
 */
#pragma once

#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/epilogue/thread/linear_combination_gelu.h"

#include "./library_ext.h"

namespace cutlass {
namespace library {

/*! \brief Epilogue kind corresponding to epilogue operator classes */
template <typename T>
struct EpilogueOpMap {
  static EpilogueKindExt const kId = EpilogueKindExt::kUnknown;
};

template <typename ElementOutput_, int Count, typename ElementAccumulator_,
          typename ElementCompute_, cutlass::epilogue::thread::ScaleType::Kind Scale,
          FloatRoundStyle Round>
struct EpilogueOpMap<cutlass::epilogue::thread::LinearCombination<
    ElementOutput_, Count, ElementAccumulator_, ElementCompute_, Scale, Round>> {
  static EpilogueKindExt const kId = EpilogueKindExt::kLinearCombination;
};

template <typename ElementOutput_, int Count, typename ElementAccumulator_,
          typename ElementCompute_, cutlass::epilogue::thread::ScaleType::Kind Scale,
          FloatRoundStyle Round>
struct EpilogueOpMap<cutlass::epilogue::thread::LinearCombinationRelu<
    ElementOutput_, Count, ElementAccumulator_, ElementCompute_, Scale, Round>> {
  static EpilogueKindExt const kId = EpilogueKindExt::kLinearCombinationRelu;
};

template <typename ElementOutput_, int Count, typename ElementAccumulator_,
          typename ElementCompute_, FloatRoundStyle Round>
struct EpilogueOpMap<cutlass::epilogue::thread::LinearCombinationGELU<
    ElementOutput_, Count, ElementAccumulator_, ElementCompute_, Round>> {
  static EpilogueKindExt const kId = EpilogueKindExt::kLinearCombinationGelu;
};

}  // namespace library
}  // namespace cutlass
