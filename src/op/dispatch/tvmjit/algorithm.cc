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

}  // namespace tvmjit
}  // namespace op
}  // namespace mnm
