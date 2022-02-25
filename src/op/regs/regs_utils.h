/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/regs/regs_utils.h
 * \brief Helpers for operator registry
 */
#pragma once
#include <vector>
#include <string>
#include <algorithm>
#include "raf/op.h"
#include "raf/ir.h"
#include "raf/registry.h"
#include "raf/value.h"
#include "raf/binding.h"
#include "raf/executor.h"

namespace raf {
namespace op {
namespace regs {

constexpr int MAX_NUM_ARGS = 256;

inline void FillError(const dmlc::Error& e, const std::string& from, const std::string& to) {
  size_t index = 0;
  std::string str = e.what();
  while (true) {
    index = str.find(from, index);
    if (index == std::string::npos) {
      break;
    }
    str.replace(index, from.length(), to);
    index += to.length();
  }
  throw dmlc::Error(str);
}

inline std::string ToOrdinal(int x) {
  if (x == 0) {
    return "first";
  }
  if (x == 1) {
    return "second";
  }
  if (x == 2) {
    return "third";
  }
  return std::to_string(x + 1) + "-th";
}

inline std::string GetTypeStr(const registry::TVMArgValue& a) {
  if (a.type_code() == kTVMObjectHandle) {
    return (a.operator ir::ObjectRef())->GetTypeKey();
  }
  return tvm::runtime::ArgTypeCode2Str(a.type_code());
}

inline bool RemoveNoGrad(binding::GradTape* tapes, ir::Expr* grads, int* n) {
  // returns: whether full grad is used
  int m = 0;
  bool full_grads = true;
  for (int i = 0, len = *n; i < len; ++i) {
    const auto& tape = tapes[i];
    const auto& grad = grads[i];
    if (tape.defined() && grad.defined()) {
      grads[m] = grads[i];
      tapes[m] = tape;
      ++m;
      continue;
    }
    if (grad.defined()) {
      full_grads = false;
    }
  }
  *n = m;
  return full_grads;
}

void CollectVars(const ir::Expr& expr, std::vector<const ir::ExprNode*>* vars);

}  // namespace regs
}  // namespace op
}  // namespace raf

namespace raf {
namespace op {
namespace regs {

class VarPack {
 public:
  ir::Var y = ir::MakeVar("y", {});
  ir::Var dy = ir::MakeVar("dy", {});
  std::array<ir::Var, MAX_NUM_ARGS> x;

  VarPack() {
    for (int i = 0; i < MAX_NUM_ARGS; ++i) {
      x[i] = ir::MakeVar("x" + std::to_string(i), {});
    }
  }

  ir::Call MakeCall(const ir::Op& op, int n) const {
    CHECK_LE(n, MAX_NUM_ARGS);
    std::vector<ir::Expr> xs(n);
    for (int i = 0; i < n; ++i) {
      xs[i] = this->x[i];
    }
    return ir::Call(op, xs);
  }

  static VarPack* Get() {
    static VarPack* pack = new VarPack();
    return pack;
  }
};

template <const char* op_name, int n_args>
struct OpPack {
  static_assert(n_args <= MAX_NUM_ARGS, "Too many arguments");
  ir::Op op = ir::Op::Get(op_name);
  value::OpValue opv = value::OpValue::make(op);
  ir::Array<ir::Expr> grads;
  std::vector<const ir::ExprNode*> grad_used_vars;
  OpPack() {
    auto fpg = ir::Op::GetAttrMap<FPrimalGradient>("FPrimalGradient");
    const auto* pack = VarPack::Get();
    if (fpg.count(op)) {
      auto call = pack->MakeCall(op, n_args);
      grads = fpg[op](call, call->args, pack->y, pack->dy);
      std::vector<ir::Expr> grads_defined;
      for (const ir::Expr& grad : grads) {
        if (grad.defined()) {
          grads_defined.push_back(grad);
        }
      }
      CollectVars(ir::Tuple(grads_defined), &grad_used_vars);
    }
  }
  static OpPack<op_name, n_args>* Get() {
    static auto* inst = new OpPack<op_name, n_args>();
    return inst;
  }
};

}  // namespace regs
}  // namespace op
}  // namespace raf
