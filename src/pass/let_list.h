/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file let_list.h
 * \brief taken from tvm relay. author: Marisa
 */
#pragma once

#include <tvm/relay/expr.h>
#include <tvm/relay/analysis.h>
#include <utility>
#include <vector>
#include <tuple>
#include <string>
#include "tvm/relay/type.h"

namespace raf {
namespace pass {

using namespace raf::ir;

/*!
 * \brief LetList allow you to transform expression into variables, so you can copy them around.
 *  one can insert into the LetList by calling Push, and wrap an expression with bindings with Get.
 *  additionally, there is the 'With' function, which automatically call Get.
 */
class LetList {
 public:
  ~LetList() {
    if (lets_.size() > 0 && !used_) {
      LOG(WARNING) << "letlist not used";
    }
  }
  /*!
   * \brief insert a binding.
   *
   * \param pv the var of the binding.
   *
   * \param expr the value of the binding.
   *
   * \return a Var that hold the inserted expr.
   */
  Var Push(Var pv, Expr expr) {
    CHECK(!used_);
    CHECK(tvm::relay::WellFormed(expr));
    lets_.emplace_back(std::make_pair(pv, expr));
    return pv;
  }

  /*!
   * \brief insert a binding.
   *
   * \param expr the value of the binding.
   *
   * \param ty the type of the binding.
   *
   * \return a Var that hold the inserted expr.
   */
  Var Push(Expr expr, Type ty) {
    std::string fullname = "x_" + std::to_string(label_++);
    return Push(raf::ir::MakeVar(fullname, ty), expr);
  }

  /*!
   * \brief insert a binding.
   *
   *  \param expr the value of the binding.
   *
   *  \return a Var that hold the inserted expr.
   */
  Var Push(Expr expr) {
    return Push(expr, Type());
  }

  /*!
   * \brief wrap an expr around the LetList.
   *
   *  \param body the Expression to be wrapped around.
   *
   *  \return the wrapped expr.
   */
  Expr Get(const Expr& body) {
    CHECK(!used_);
    Expr ret = body;
    for (auto rit = lets_.rbegin(); rit != lets_.rend(); ++rit) {
      ret = raf::ir::Let(std::get<0>(*rit), std::get<1>(*rit), ret);
    }
    used_ = true;
    return ret;
  }

  /*! \brief generate an LetList and wrap the result automatically.
   *
   *  \param f a function that generate the unwrapped Expr.
   *
   *  \code
   *  // Example code that generate `16 * a` using 4 plus instead of 15 plus.
   *  Expr mult_sixteen(const Var& a) {
   *    Op plus = Op::Get("plus");
   *    // Automatically call Get with LetList::With
   *    return LetList::With([&](LetList* ll) {
   *      // Turn a call to plus into a variable to avoid duplication of code
   *      Var b = ll->Push(Call(plus, {a, a}));
   *      Var c = ll->Push(Call(plus, {b, b}));
   *      Var d = ll->Push(Call(plus, {c, c}));
   *      return Call(plus, {d, d});
   *    });
   *  }
   *  \endcode
   *
   *  \return the wrapped Expr.
   */
  template <typename F>
  static Expr With(F&& f) {
    LetList ll;
    return ll.Get(f(&ll));
  }

  static Expr Let(const Expr& e, const std::function<Expr(const Var&)>& f) {
    return With([&](LetList* ll) { return f(ll->Push(e)); });
  }

 private:
  std::vector<std::pair<Var, Expr> > lets_;
  int64_t label_ = 0;
  bool used_ = false;
};

}  // namespace pass
}  // namespace raf
