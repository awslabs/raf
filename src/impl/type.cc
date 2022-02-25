/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/impl/type.cc
 * \brief Type system
 */
#include "tvm/runtime/memory.h"
#include "raf/type.h"

namespace raf {
namespace ir {

OpType MakeOpType(const std::string& op_name, const std::string& fn_name,
                  tvm::runtime::TypedPackedFunc<tvm::relay::Type(const op::CallValues& value)> fn) {
  auto func_name = std::string("raf.type.type_inference.") + fn_name;
  TypeInferenceFn env_fn;

  if (tvm::runtime::Registry::Get(func_name)) {
    env_fn = tvm::EnvFunc::Get(func_name);
  } else {
    tvm::runtime::Registry::Register(func_name).set_body(fn);
    env_fn = tvm::EnvFunc::Get(func_name);
  }

  const Op& op = tvm::OpRegEntry::RegisterOrGet(op_name).op();
  Array<TypeVar> type_params;
  Array<Type> arg_types;

  // Add inputs
  std::string input_name_prefix = fn_name + "_in";
  for (int i = 0; i < op->num_inputs; ++i) {
    auto name = input_name_prefix + std::to_string(i);
    auto param = TypeVar(name, tvm::TypeKind::kType);
    type_params.push_back(param);
    arg_types.push_back(param);
  }

  // Add output type.
  auto out_param = TypeVar(fn_name + "_out", tvm::TypeKind::kType);
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

RAF_REGISTER_OBJECT_REFLECT(TypeInferenceNode);

}  // namespace ir
}  // namespace raf
