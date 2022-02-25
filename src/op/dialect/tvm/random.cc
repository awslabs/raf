/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file ./src/op/dialect/tvm/random.cc
 * \brief Random operators bridged from TVM.
 */
#include <vector>
#include "raf/value.h"
#include "./tvm_attrs.h"
#include "./tvm_utils.h"
#include "../../schema/random.h"

namespace raf {
namespace op {
namespace tvm_dialect {

using namespace raf::ir;
using namespace raf::value;
using namespace raf::tensor;
using namespace raf::op::schema;

std::vector<Value> ThreefryGenerateSchema2Args(const ThreefryGenerateArgs* args) {
  return {args->key};
}

std::vector<std::string> ThreefryGenerateSchemaArgNames(const op::CallValues& call) {
  return {"key"};
}

Attrs ThreefryGenerateSchema2Attrs(const ThreefryGenerateArgs* args) {
  auto attrs = make_object<ThreefryGenerateAttrs>();
  std::vector<IndexExpr> shape;
  shape.reserve(args->shape.size());
  for (size_t i = 0; i < args->shape.size(); ++i) {
    shape.emplace_back(IntImm(ir::DataType::Int(32), args->shape[i]));
  }
  attrs->out_shape = Array<Integer>(shape.begin(), shape.end());
  return Attrs(attrs);
}

HashKey ThreefryGenerateHasher(const std::vector<Type>& param_types, const Type& y_type,
                               const ThreefryGenerateArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->shape;
  return key;
}

RAF_TVM(threefry_generate, ThreefryGenerate, ThreefryGenerateArgs, ThreefryGenerateSchema2Args,
        ThreefryGenerateSchemaArgNames, ThreefryGenerateSchema2Attrs, ThreefryGenerateHasher,
        kOpaque);

std::vector<Value> ThreefrySplitSchema2Args(const ThreefrySplitArgs* args) {
  return {args->key};
}

std::vector<std::string> ThreefrySplitSchemaArgNames(const op::CallValues& call) {
  return {"key"};
}

RAF_TVM(threefry_split, ThreefrySplit, ThreefrySplitArgs, ThreefrySplitSchema2Args,
        ThreefrySplitSchemaArgNames, GenericAttrs, GenericHasher, kOpaque);

}  // namespace tvm_dialect
}  // namespace op
}  // namespace raf
