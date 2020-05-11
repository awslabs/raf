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

DLTensor GetDLTensor(const Value& v) {
  DLTensor* ret = v;
  return *ret;
}

void GetOut(const Value& out, std::vector<DLTensor>* ret) {
  CHECK(ret->empty());
  if (out->IsInstance<TensorValueObj>()) {
    DLTensor* t = out;
    ret->emplace_back(*t);
  } else if (const auto* tv = out.as<TupleValueObj>()) {
    for (const auto& v : tv->fields) {
      DLTensor* t = v;
      ret->emplace_back(*t);
    }
  } else {
    LOG(FATAL) << "InternalError: TVMOpEnv does not deal with " << out->GetTypeKey();
    throw;
  }
}

Type GetTensorType(const DLTensor& dlt) {
  auto shape = GetShape<Integer>(dlt);
  return TensorTypeNode::make({shape.begin(), shape.end()}, tvm::relay::DataType(dlt.dtype));
}

Type GetTupleType(const std::vector<DLTensor>& dlts) {
  std::vector<Type> types;
  for (const auto& dlt : dlts) {
    types.emplace_back(GetTensorType(dlt));
  }
  return TupleTypeNode::make(types);
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

PackedFunc CompileOp(const Op& op,                          //
                     const Attrs& attrs,                    //
                     const std::vector<Type>& param_types,  //
                     const Type& ret_type,                  //
                     Context ctx) {
  static auto engine = GetPackedFunc("relay.backend._CompileEngineGlobal")();
  static auto c_cache_key = GetPackedFunc("relay.backend._make_CCacheKey");
  static auto jit = GetPackedFunc("relay.backend._CompileEngineJIT");
  static auto engine_clear = GetPackedFunc("relay.backend._CompileEngineClear");
  Function func;
  {
    std::vector<Var> params;
    for (int i = 0, n = param_types.size(); i < n; ++i) {
      auto var = VarNode::make("", param_types[i]);
      var->checked_type_ = param_types[i];
      params.push_back(var);
    }
    func = FunctionNode::make(params,                                                     //
                              CallNode::make(op, {params.begin(), params.end()}, attrs),  //
                              ret_type,                                                   //
                              {});
    func->body->checked_type_ = ret_type;
    func->checked_type_ = FuncTypeNode::make(param_types, ret_type, {}, {});
  }
  tvm::Target target;
  {
    if (ctx.device_type == DevType::kCPU()) {
      target = tvm::target::llvm();
    } else if (ctx.device_type == DevType::kCUDA()) {
      target = tvm::target::cuda();
    } else {
      LOG(FATAL) << "NotImplementedError: target is not supported " << ctx.device_type.c_str();
      throw;
    }
  }
  engine_clear(engine);
  return jit(engine, c_cache_key(func, target));
}

void TVMOpEnv::Setup() {
  for (auto& dlt : outputs) {
    RequestMemory(&dlt.data, dlt.ctx, BytesCompactTensor(dlt));
  }
  SetArgs(&inputs, &outputs, &values, &codes);
}

void TVMOpEnv::Execute(const op::CallValues& call) {
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

}  // namespace tvmjit
}  // namespace op
}  // namespace mnm
