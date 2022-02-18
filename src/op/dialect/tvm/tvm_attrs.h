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
 * \file ./src/op/tvm/tvm_attrs.h
 * \brief Attributes defined in TVM
 */
#pragma once
#include <tvm/ir/attrs.h>
#include <tvm/relay/attrs/algorithm.h>
#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/attrs/device_copy.h>
#include <tvm/relay/attrs/image.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/random.h>
#include <tvm/relay/attrs/reduce.h>
#include <tvm/relay/attrs/transform.h>
#include "mnm/ir.h"
#include "mnm/value.h"
#include "mnm/op.h"
#include "./attrs/nn.h"
#include "./attrs/optimizer.h"
#include "./attrs/reduce.h"
#include "./attrs/transform.h"
#include "./attrs/unary.h"

namespace mnm {
namespace op {
namespace tvm_dialect {

// Reuse attrs from TVM
using tvm::relay::ArangeAttrs;
using tvm::relay::ArgReduceAttrs;
using tvm::relay::ArgsortAttrs;
using tvm::relay::BiasAddAttrs;
using tvm::relay::CastAttrs;
using tvm::relay::ClipAttrs;
using tvm::relay::CompilerAttrs;
using tvm::relay::ConcatenateAttrs;
using tvm::relay::Conv2DAttrs;
using tvm::relay::DeviceCopyAttrs;
using tvm::relay::DropoutAttrs;
using tvm::relay::ExpandDimsAttrs;
using tvm::relay::GatherAttrs;
using tvm::relay::GatherNDAttrs;
using tvm::relay::InitOpAttrs;
using tvm::relay::OneHotAttrs;
using tvm::relay::ReduceAttrs;
using tvm::relay::RepeatAttrs;
using tvm::relay::ReshapeAttrs;
using tvm::relay::Resize2DAttrs;
using tvm::relay::ReverseAttrs;
using tvm::relay::ReverseSequenceAttrs;
using tvm::relay::ScatterAttrs;
using tvm::relay::SequenceMaskAttrs;
using tvm::relay::SplitAttrs;
using tvm::relay::SqueezeAttrs;
using tvm::relay::StackAttrs;
using tvm::relay::StridedSliceAttrs;
using tvm::relay::TakeAttrs;
using tvm::relay::ThreefryGenerateAttrs;
using tvm::relay::TopKAttrs;
using tvm::relay::TransposeAttrs;

// Helper functions
template <typename T>
std::vector<value::Value> BinarySchema2Args(const T* args) {
  return {args->x1, args->x2};
}

inline std::vector<std::string> BinarySchemaArgNames(const op::CallValues& call) {
  return {"x1", "x2"};
}

template <typename T>
tvm::Attrs GenericAttrs(const T* args) {
  return tvm::Attrs();
}

}  // namespace tvm_dialect
}  // namespace op
}  // namespace mnm
