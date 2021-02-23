/*!
 * Copyright (c) 2021 by Contributors
 * \file ./src/op/dispatch/tvmjit/init.cc
 * \brief Init operators bridged from TVM.
 */
#include <tvm/relay/attrs/transform.h>
#include <mnm/value.h>
#include <array>
#include "./tvmjit_utils.h"
#include "../../schema/init.h"

namespace mnm {
namespace op {
namespace tvmjit {

using namespace mnm::ir;
using namespace mnm::value;
using namespace mnm::tensor;
using namespace mnm::op::schema;
using namespace tvm;
using namespace ::tvm::relay;

std::vector<Value> InitOpSchema2Args(const InitOpArgs* args) {
  return {};
}

std::vector<std::string> InitOpSchemaArgNames(const op::CallValues& call) {
  return {};
}

Attrs InitOpSchema2Attrs(const InitOpArgs* args) {
  auto attrs = make_object<InitOpAttrs>();
  std::vector<IndexExpr> shape;
  shape.reserve(args->shape.size());
  for (size_t i = 0; i < args->shape.size(); ++i) {
    shape.emplace_back(IntImm(ir::DataType::Int(32), args->shape[i]));
  }
  attrs->shape = Array<Integer>(shape.begin(), shape.end());
  attrs->dtype = DataType(ir::String2DLDataType(args->dtype));
  return Attrs(attrs);
}

HashKey InitOpHasher(const std::vector<Type>& param_types, const Type& y_type,
                     const InitOpArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->shape;
  key << ir::String2DLDataType(args->dtype);
  key << args->device;
  return key;
}

MNM_TVMJIT(Zeros, "mnm.op.zeros", InitOpArgs, InitOpSchema2Args, InitOpSchemaArgNames,
           InitOpSchema2Attrs, InitOpHasher);
MNM_TVMJIT(Ones, "mnm.op.ones", InitOpArgs, InitOpSchema2Args, InitOpSchemaArgNames,
           InitOpSchema2Attrs, InitOpHasher);

std::vector<Value> OneHotSchema2Args(const OneHotArgs* args) {
  return {args->indices, args->on_value, args->off_value};
}

std::vector<std::string> OneHotSchemaArgNames(const op::CallValues& call) {
  return {"indices", "on_value", "off_value"};
}

Attrs OneHotSchema2Attrs(const OneHotArgs* args) {
  auto attrs = make_object<OneHotAttrs>();
  attrs->depth = args->depth;
  attrs->axis = args->axis;
  attrs->dtype = DataType(ir::String2DLDataType(args->dtype));
  return Attrs(attrs);
}

HashKey OneHotHasher(const std::vector<Type>& param_types, const Type& y_type,
                     const OneHotArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->depth;
  key << args->axis;
  key << ir::String2DLDataType(args->dtype);
  key << args->device;
  return key;
}

MNM_TVMJIT(OneHot, "mnm.op.one_hot", OneHotArgs, OneHotSchema2Args, OneHotSchemaArgNames,
           OneHotSchema2Attrs, OneHotHasher);

}  // namespace tvmjit
}  // namespace op
}  // namespace mnm
