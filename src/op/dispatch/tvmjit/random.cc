/*!
 * Copyright (c) 2021 by Contributors
 * \file ./src/op/dispatch/tvmjit/random.cc
 * \brief Random operators bridged from TVM.
 */
#include <tvm/relay/attrs/random.h>
#include <mnm/value.h>
#include <array>
#include "./tvm_attrs.h"
#include "./tvmjit_utils.h"
#include "../../schema/random.h"
namespace mnm {
namespace op {
namespace tvmjit {

using namespace mnm::ir;
using namespace mnm::value;
using namespace mnm::tensor;
using namespace mnm::op::schema;
using namespace tvm;
using namespace tvm::relay;

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

MNM_TVMJIT(ThreefryGenerate, "mnm.op.threefry_generate", ThreefryGenerateArgs,
           ThreefryGenerateSchema2Args, ThreefryGenerateSchemaArgNames,
           ThreefryGenerateSchema2Attrs, ThreefryGenerateHasher);

std::vector<Value> ThreefrySplitSchema2Args(const ThreefrySplitArgs* args) {
  return {args->key};
}

std::vector<std::string> ThreefrySplitSchemaArgNames(const op::CallValues& call) {
  return {"key"};
}

MNM_TVMJIT(ThreefrySplit, "mnm.op.threefry_split", ThreefrySplitArgs, ThreefrySplitSchema2Args,
           ThreefrySplitSchemaArgNames, GenericAttrs, GenericHasher);

}  // namespace tvmjit
}  // namespace op
}  // namespace mnm
