/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file ./src/op/dialect/tvm/tvm_utils.cc
 * \brief Implementation of utility methods for TVM dialect.
 */
#include "mnm/value.h"
#include "mnm/registry.h"
#include "./tvm_utils.h"
#include "../../../common/shape_utils.h"

namespace mnm {
namespace op {
namespace tvm_dialect {

using namespace mnm::value;
using namespace mnm::ir;
using namespace mnm::registry;
using common::shape_utils::BytesCompactTensor;
using common::shape_utils::GetShape;

MetaPersistCache<TVMModuleCacheEntry> CacheBuildCpu("tvm_cpu");
MetaPersistCache<TVMModuleCacheEntry> CacheBuildCuda("tvm_cuda");
MetaPersistCache<RelayFuncCacheEntry> CacheLoweredFunc("tvm_lower");

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

PackedMetricMap DumpTVMCacheMetric(const std::string& cache_name) {
  static std::unordered_map<std::string, MetaCacheMetric*> name_to_cache = {
      {"tvm_cpu", &CacheBuildCpu},
      {"tvm_cuda", &CacheBuildCuda},
      {"tvm_lower", &CacheLoweredFunc},
  };

  PackedMetricMap ret;
  if (name_to_cache.count(cache_name) == 0) {
    LOG(WARNING) << "Cannot find cache " << cache_name << " for dumping metric";
    return ret;
  }

  auto metrics = name_to_cache[cache_name]->GetMetric();
  for (const auto& it : metrics) {
    ret.Set(it.first, it.second);
  }
  return ret;
}

MNM_REGISTER_GLOBAL("mnm.cache.DumpTVMCacheMetric").set_body_typed(DumpTVMCacheMetric);

MNM_REGISTER_DIALECT("tvm").set_enable(DevType::kCPU()).set_enable(DevType::kCUDA());
TVM_REGISTER_PASS_CONFIG_OPTION("mnm.tvm.allow_jit_failure", tvm::Bool);

}  // namespace tvm_dialect
}  // namespace op
}  // namespace mnm
