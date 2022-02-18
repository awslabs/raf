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
 * \file src/op/dialect/cutlass/timer.cc
 * \brief Timer for cutlass kernel
 */
#include "./timer.h"

namespace mnm {
namespace op {
namespace cutlass {

using namespace mnm::registry;
using namespace mnm::value;

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
}  // namespace mnm
