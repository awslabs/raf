/*!
 * Copyright (c) 2019 by Contributors
 * \file ./src/op/dispatch/tvmjit/vision.cc
 * \brief NN-related operators bridged from TVM.
 */
#include <tvm/relay/attrs/vision.h>
#include <mnm/value.h>
#include <array>
#include "./tvmjit_utils.h"
#include "./tvm_attrs.h"
#include "../../schema/transform.h"
#include "../../schema/nn.h"
#include "../../schema/vision.h"
#include "../../../common/shape_utils.h"

namespace mnm {
namespace op {
namespace tvmjit {

using namespace mnm::ir;
using namespace mnm::value;
using namespace mnm::op::schema;
using namespace tvm;
using namespace ::tvm::relay;

Attrs GetValidCountsNormalizer(TVMOpEnv* env, const GetValidCountsArgs* args) {
  CHECK_EQ(env->outputs.size(), 3U);
  env->inputs.resize(1);
  env->inputs[0] = GetDLTensor(args->data);
  auto attrs = make_object<GetValidCountsAttrs>();
  attrs->score_threshold = args->score_threshold;
  attrs->id_index = args->id_index;
  attrs->score_index = args->score_index;
  return Attrs(attrs);
}

void GetValidCountsTyper(TVMOpEnv* env, std::vector<Type>* param_types, Type* y_type) {
  *y_type = GetTupleType(env->outputs);
  *param_types = {GetTensorType(env->inputs[0])};
}

HashKey GetValidCountsHasher(const std::vector<Type>& param_types, const Type& y_type,
                             const GetValidCountsArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->score_threshold;
  key << args->id_index;
  key << args->score_index;
  return key;
}

MNM_TVMJIT(GetValidCounts, "mnm.op.get_valid_counts", GetValidCountsArgs, GetValidCountsNormalizer,
           GetValidCountsTyper, GetValidCountsHasher);

}  // namespace tvmjit
}  // namespace op
}  // namespace mnm
