/*!
 * Copyright (c) 2020 by Contributors
 * \file src/impl/type.cc
 * \brief Type system
 */
#include "tvm/runtime/memory.h"
#include "mnm/type.h"

namespace mnm {
namespace type {

using namespace mnm::ir;

OpType MakeOpType(const std::string& op_name, const std::string& fn_name,
                  tvm::runtime::TypedPackedFunc<tvm::relay::Type(const op::CallValues& value)> fn) {
  using namespace tvm;
  auto func_name = std::string("mnm.type.type_inference.") + fn_name;
  TypeInferenceFn env_fn;

  if (runtime::Registry::Get(func_name)) {
    env_fn = EnvFunc::Get(func_name);
  } else {
    runtime::Registry::Register(func_name).set_body(fn);
    env_fn = EnvFunc::Get(func_name);
  }

  const Op& op = tvm::OpRegEntry::RegisterOrGet(op_name).op();
  Array<TypeVar> type_params;
  Array<Type> arg_types;

  // Add inputs
  std::string input_name_prefix = fn_name + "_in";
  for (int i = 0; i < op->num_inputs; ++i) {
    auto name = input_name_prefix + std::to_string(i);
    auto param = TypeVar(name, TypeKind::kType);
    type_params.push_back(param);
    arg_types.push_back(param);
  }

  // Add output type.
  auto out_param = TypeVar(fn_name + "_out", TypeKind::kType);
  type_params.push_back(out_param);

  TypeConstraint type_inference = TypeInference(env_fn);

  auto func_type = FuncType(arg_types, out_param, type_params, {type_inference});

  op->op_type = func_type;

  return func_type;
}

TypeInference::TypeInference(TypeInferenceFn func) {
  ObjectPtr<TypeInferenceNode> n = make_object<TypeInferenceNode>();
  n->func = std::move(func);
  data_ = std::move(n);
}

MNM_REGISTER_OBJECT_REFLECT(TypeInferenceNode);

}  // namespace type
}  // namespace mnm
