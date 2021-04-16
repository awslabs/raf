/*!
 * Copyright (c) 2021 by Contributors
 * \file include/cutlass_ext/library/library_internal_ext.h
 * \brief Extentions for cutlass library internal classes
 */
#pragma once

#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/epilogue/thread/linear_combination_gelu.h"

namespace cutlass {
namespace library {

/*! \brief Epilogue kind corresponding to epilogue operator classes */
template <typename T>
struct EpilogueOpMap {
  static EpilogueKind const kId = EpilogueKind::kUnknown;
};

template <typename ElementOutput_, int Count, typename ElementAccumulator_,
          typename ElementCompute_, cutlass::epilogue::thread::ScaleType::Kind Scale,
          FloatRoundStyle Round>
struct EpilogueOpMap<cutlass::epilogue::thread::LinearCombination<
    ElementOutput_, Count, ElementAccumulator_, ElementCompute_, Scale, Round>> {
  static EpilogueKind const kId = EpilogueKind::kLinearCombination;
};

template <typename ElementOutput_, int Count, typename ElementAccumulator_,
          typename ElementCompute_, cutlass::epilogue::thread::ScaleType::Kind Scale,
          FloatRoundStyle Round>
struct EpilogueOpMap<cutlass::epilogue::thread::LinearCombinationRelu<
    ElementOutput_, Count, ElementAccumulator_, ElementCompute_, Scale, Round>> {
  static EpilogueKind const kId = EpilogueKind::kLinearCombinationRelu;
};

}  // namespace library
}  // namespace cutlass
