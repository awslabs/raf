/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/dialect/cutlass/timer.h
 * \brief Timer for cutlass kernel
 */
#pragma once

#include "./cutlass_utils.h"

#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

namespace raf {
namespace op {
namespace cutlass {

/*! \brief Module with a single PackedFunc */
class CutlassModuleNode : public tvm::runtime::ModuleNode {
 public:
  explicit CutlassModuleNode(registry::PackedFunc pf) : pf_(pf) {
  }

  virtual const char* type_key() const final {
    return "CutlassModule";
  }

  virtual registry::PackedFunc GetFunction(const std::string& name,
                                           const ir::ObjectPtr<ir::Object>& sptr_to_self) final {
    return pf_;
  }

 private:
  /*! \brief The packed function contained by this module */
  registry::PackedFunc pf_;
};

/*! \brief Wraps a packed function with CutlassModule
 *  \param pf The packed function to be wrapped
 *  \return The CutlassModule
 */
tvm::runtime::Module MakeCutlassModule(registry::PackedFunc pf);

/*! \brief Evaluates the running time of a packed function
 *  \param pf The packed function to be evaluated
 *  \param dev The execution device
 *  \param number The number of times to run this function for taking average.
 *                We call these runs as one `repeat` of measurement.
 *  \param repeat The number of times to repeat the measurement.
 *                In total, the function will be invoked (1 + number x repeat) times,
 *                where the first one is warm up and will be discarded.
 *                The returned result contains `repeat` costs,
 *                each of which is an average of `number` costs.
 *  \param min_repeat_ms The minimum duration of one `repeat` in milliseconds.
 *                       By default, one `repeat` contains `number` runs. If this parameter is set,
 *                       the parameters `number` will be dynamically adjusted to meet the
 *                       minimum duration requirement of one `repeat`.
 *                       i.e., When the run time of one `repeat` falls below this time, the `number`
 *                       parameter will be automatically increased.
 * \param limit_zero_time_iterations The maximum number of repeats when
 *                                   measured time is equal to 0. It helps to avoid hanging
 *                                   during measurements.
 * \param cooldown_interval_ms The cooldown interval in milliseconds between the number of
 *                             repeats defined by `repeats_to_cooldown`.
 * \param repeats_to_cooldown The number of repeats before the cooldown is activated.
 *  \return The function that takes same argument as func
 *          and returns Array<FloatValue>, which reports `repeat` time costs in seconds.
 */
registry::PackedFunc TimeEvaluator(registry::PackedFunc pf, Device dev, int number = 10,
                                   int repeat = 1, int min_repeat_ms = 0,
                                   int limit_zero_time_iterations = 100,
                                   int cooldown_interval_ms = 0, int repeats_to_cooldown = 1);

}  // namespace cutlass
}  // namespace op
}  // namespace raf
