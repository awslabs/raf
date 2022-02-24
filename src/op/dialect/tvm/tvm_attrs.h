/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
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
#include "raf/ir.h"
#include "raf/value.h"
#include "raf/op.h"
#include "./attrs/nn.h"
#include "./attrs/optimizer.h"
#include "./attrs/reduce.h"
#include "./attrs/transform.h"
#include "./attrs/unary.h"

namespace raf {
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
}  // namespace raf
