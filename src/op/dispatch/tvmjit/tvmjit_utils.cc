/*!
 * Copyright (c) 2019 by Contributors
 * \file ./src/op/dispatch/tvmjit/tvmjit_utils.cc
 * \brief Implementation of utility methods for TVM JIT.
 */
#include "mnm/value.h"
#include "mnm/registry.h"
#include "./tvmjit_utils.h"
#include "../../../common/shape_utils.h"

namespace mnm {
namespace op {
namespace tvmjit {

using namespace mnm::value;
using namespace mnm::ir;
using namespace mnm::registry;
using common::shape_utils::BytesCompactTensor;
using common::shape_utils::GetShape;

void GetDLTensor(const Value& v, std::vector<DLTensor>* tensors) {
  if (v->IsInstance<TensorValueObj>()) {
    DLTensor* t = v;
    tensors->emplace_back(*t);
  } else if (const auto* tv = v.as<TupleValueObj>()) {
    for (const auto& v : tv->fields) {
      DLTensor* t = v;
      tensors->emplace_back(*t);
    }
  } else {
    LOG(FATAL) << "InternalError: TVMOpEnv does not deal with " << v->GetTypeKey();
    throw;
  }
}

Type GetTensorType(const DLTensor& dlt) {
  auto shape = GetShape<Integer>(dlt);
  return TensorType({shape.begin(), shape.end()}, ir::DataType(dlt.dtype));
}

Type GetTupleType(const std::vector<DLTensor>& dlts) {
  std::vector<Type> types;
  for (const auto& dlt : dlts) {
    types.emplace_back(GetTensorType(dlt));
  }
  return TupleType(types);
}

Function LowerOp(const Op& op, const Attrs& attrs, const std::vector<Type>& param_types,
                 const Type& ret_type) {
  Function func;
  std::vector<Var> params;
  for (int i = 0, n = param_types.size(); i < n; ++i) {
    auto var = mnm::ir::MakeVar("", param_types[i]);
    var->checked_type_ = param_types[i];
    params.push_back(var);
  }
  func = Function(params, Call(op, {params.begin(), params.end()}, attrs), ret_type, {});
  func->body->checked_type_ = ret_type;
  func->checked_type_ = FuncType(param_types, ret_type, {}, {});
  return func;
}

std::string GetUniqueName(std::string name) {
  static std::unordered_map<std::string, int> name_map;
  for (size_t i = 0; i < name.length(); ++i) {
    if (name[i] == '.') name[i] = '_';
  }
  while (true) {
    auto it = name_map.find(name);
    if (it == name_map.end()) {
      name_map[name] = 1;
      return name;
    } else {
      std::ostringstream os;
      os << name << "_" << it->second;
      ++(it->second);
      name = os.str();
    }
  }
  return name;
}

std::string TruncateName(std::string name) {
  constexpr static size_t kMaxFuncNameLength = 80;
  if (name.size() > kMaxFuncNameLength) {
    std::stringstream truncated_name;
    truncated_name << name.substr(0, kMaxFuncNameLength);
    truncated_name << "_" << std::hash<std::string>{}(name) << "_";
    name = truncated_name.str();
  }
  return name;
}

void SetArgs(std::vector<DLTensor>* i, std::vector<DLTensor>* o, std::vector<TVMValue>* values,
             std::vector<int>* codes) {
  int arity = i->size() + o->size();
  values->resize(arity);
  codes->resize(arity);
  TVMArgsSetter setter(values->data(), codes->data());
  int cnt = 0;
  for (DLTensor& dlt : *i) {
    setter(cnt++, &dlt);
  }
  for (DLTensor& dlt : *o) {
    setter(cnt++, &dlt);
  }
}

void TVMOpEnv::Execute(const op::CallValues& call) {
  std::vector<TVMValue> values;
  std::vector<int> codes;
  SetArgs(&inputs, &outputs, &values, &codes);
  TVMArgs targs(values.data(), codes.data(), values.size());
  TVMRetValue rv;
  f.CallPacked(targs, &rv);
  if (call->out->IsInstance<TensorValueObj>()) {
    DLTensor* dlt = Downcast<value::TensorValue>(call->out);
    dlt->data = outputs[0].data;
  } else if (const auto* tv = call->out.as<value::TupleValueObj>()) {
    int i = 0;
    for (const auto& v : tv->fields) {
      DLTensor* dlt = Downcast<value::TensorValue>(v);
      dlt->data = outputs[i++].data;
    }
  } else {
    LOG(FATAL) << "InternalError: internal error.";
    throw;
  }
}

void TVMOpEnv::Execute(const std::vector<Value>& inputs, Value output) {
  this->inputs.clear();
  this->outputs.clear();
  for (auto val : inputs) {
    GetDLTensor(val, &this->inputs);
  }
  GetDLTensor(output, &this->outputs);
  std::vector<TVMValue> values;
  std::vector<int> codes;
  SetArgs(&this->inputs, &this->outputs, &values, &codes);
  TVMArgs targs(values.data(), codes.data(), values.size());
  TVMRetValue rv;

  // Skip the execution if we are in the task extraction mode since
  // we do not care about the correctness.
  if (AllowJitFailure()) {
    return;
  }

  f.CallPacked(targs, &rv);
}

TVM_REGISTER_PASS_CONFIG_OPTION("mnm.tvmjit.allow_jit_failure", tvm::Bool);

}  // namespace tvmjit
}  // namespace op
}  // namespace mnm
