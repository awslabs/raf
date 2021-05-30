/*!
 * Copyright (c) 2019 by Contributors
 * \file ./src/op/dispatch/tvmjit/vision.cc
 * \brief NN-related operators bridged from TVM.
 */
#include <tvm/relay/attrs/algorithm.h>
#include <mnm/value.h>
#include <array>
#include "./tvmjit_utils.h"
#include "./tvm_attrs.h"
#include "../../schema/transform.h"
#include "../../schema/nn.h"
#include "../../schema/algorithm.h"
#include "../../../common/shape_utils.h"

namespace mnm {
namespace op {
namespace tvmjit {

using namespace mnm::ir;
using namespace mnm::value;
using namespace mnm::op::schema;
using namespace tvm;
using namespace ::tvm::relay;

std::vector<Value> ArgsortSchema2Args(const ArgsortArgs* args) {
  return {args->data};
}

std::vector<std::string> ArgsortSchemaArgNames(const op::CallValues& call) {
  return {"data"};
}

Attrs ArgsortSchema2Attrs(const ArgsortArgs* args) {
  auto attrs = make_object<ArgsortAttrs>();
  attrs->axis = args->axis;
  attrs->is_ascend = args->is_ascend;
  attrs->dtype = DataType(ir::String2DLDataType(args->dtype));
  return Attrs(attrs);
}

HashKey ArgsortHasher(const std::vector<Type>& param_types, const Type& y_type,
                      const ArgsortArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->axis;
  key << args->is_ascend;
  key << ir::String2DLDataType(args->dtype);
  return key;
}

MNM_TVMJIT(Argsort, "mnm.op.argsort", ArgsortArgs, ArgsortSchema2Args, ArgsortSchemaArgNames,
           ArgsortSchema2Attrs, ArgsortHasher);

std::vector<Value> SortSchema2Args(const SortArgs* args) {
  return {args->data};
}

std::vector<std::string> SortSchemaArgNames(const op::CallValues& call) {
  return {"data"};
}

Attrs SortSchema2Attrs(const SortArgs* args) {
  auto attrs = make_object<ArgsortAttrs>();
  attrs->axis = args->axis;
  attrs->is_ascend = args->is_ascend;
  return Attrs(attrs);
}

HashKey SortHasher(const std::vector<Type>& param_types, const Type& y_type, const SortArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->axis;
  key << args->is_ascend;
  return key;
}

MNM_TVMJIT(Sort, "mnm.op.sort", SortArgs, SortSchema2Args, SortSchemaArgNames, SortSchema2Attrs,
           SortHasher);

std::vector<Value> TopkSchema2Args(const TopkArgs* args) {
  return {args->data};
}

std::vector<std::string> TopkSchemaArgNames(const op::CallValues& call) {
  return {"data"};
}

Attrs TopkSchema2Attrs(const TopkArgs* args) {
  auto attrs = make_object<TopKAttrs>();
  attrs->k = IntImm(tvm::runtime::DataType::Int(64), args->k);
  attrs->axis = args->axis;
  attrs->ret_type = args->ret_type;
  attrs->is_ascend = args->is_ascend;
  attrs->dtype = DataType(ir::String2DLDataType(args->dtype));
  return Attrs(attrs);
}

HashKey TopkHasher(const std::vector<Type>& param_types, const Type& y_type, const TopkArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->k;
  key << args->axis;
  key << args->ret_type;
  key << args->is_ascend;
  key << ir::String2DLDataType(args->dtype);
  return key;
}

MNM_TVMJIT(Topk, "mnm.op.topk", TopkArgs, TopkSchema2Args, TopkSchemaArgNames, TopkSchema2Attrs,
           TopkHasher);

}  // namespace tvmjit
}  // namespace op
}  // namespace mnm
