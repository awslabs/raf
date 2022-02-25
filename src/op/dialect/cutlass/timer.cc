/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/dialect/cutlass/timer.cc
 * \brief Timer for cutlass kernel
 */
#include "./timer.h"

namespace raf {
namespace op {
namespace cutlass {

using namespace raf::registry;
using namespace raf::value;

tvm::runtime::Module MakeCutlassModule(PackedFunc pf) {
  ObjectPtr<CutlassModuleNode> n = tvm::runtime::make_object<CutlassModuleNode>(pf);
  return tvm::runtime::Module(n);
}

PackedFunc TimeEvaluator(PackedFunc pf, Device dev, int number, int repeat, int min_repeat_ms) {
  tvm::Device tvm_dev = dev;
  auto wrapper = [=](TVMArgs args, TVMRetValue* rv) mutable {
    const static PackedFunc rpv_eval = registry::GetPackedFunc("runtime.RPCTimeEvaluator");
    PackedFunc timer = rpv_eval(MakeCutlassModule(pf), "main", (int)tvm_dev.device_type,
                                (int)tvm_dev.device_id, number, repeat, min_repeat_ms, "");
    TVMRetValue timer_rv;
    timer.CallPacked(args, &timer_rv);
    const double* speed = reinterpret_cast<const double*>(timer_rv.operator std::string().data());
    std::vector<FloatValue> ret;
    ret.reserve(repeat);
    for (int i = 0; i < repeat; ++i) {
      ret.push_back(FloatValue::make(DataType::Float(64), speed[i]));
    }
    *rv = Array<FloatValue>(ret);
  };
  return PackedFunc(wrapper);
}

}  // namespace cutlass
}  // namespace op
}  // namespace raf
