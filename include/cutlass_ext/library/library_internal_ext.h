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
