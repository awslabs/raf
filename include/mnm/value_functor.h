/*!
 * Copyright (c) 2019 by Contributors
 * \file value_functor.h
 * \brief A powerful visitor which enables defining arbitrary function
 * signatures with type based dispatch on first argument.
 */
#pragma once

#include <utility>

#include "./value.h"

namespace mnm {
namespace value {

/*!
 * \brief A dynamical functor that dispatches on in the first Value argument.
 *  You can use this as a more powerful Visitor, since it allows you to
 *  define function signatures of Visit Function.
 *
 * \tparam FType function signiture
 *  This type is only defined for FType with function signature R(const Value&,
 * Args...)
 */
template <typename FType>
class ValueFunctor;

// functions to be overriden.
#define VALUE_FUNCTOR_DEFAULT                                      \
  { return VisitValueDefault_(op, std::forward<Args>(args)...); }

#define VALUE_FUNCTOR_DISPATCH(OP)                                \
  vtable.template set_dispatch<OP>(                                    \
      [](const ir::ObjectRef& n, TSelf* self, Args... args) {                \
        return self->VisitValue_(static_cast<const OP*>(n.get()), \
                                 std::forward<Args>(args)...);          \
      });

template <typename R, typename... Args>
class ValueFunctor<R(const Value& n, Args...)> {
 private:
  using TSelf = ValueFunctor<R(const Value& n, Args...)>;
  using FType = tvm::NodeFunctor<R(const ir::ObjectRef& n, TSelf* self, Args...)>;

 public:
  /*! \brief the result type of this functor */
  using result_type = R;
  /*! \brief virtual destructor */
  virtual ~ValueFunctor() {}
  /*!
   * \brief Same as call.
   * \param n The value node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  R operator()(const Value& n, Args... args) {
    return VisitValue(n, std::forward<Args>(args)...);
  }
  /*!
   * \brief The functor call.
   * \param n The value node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  virtual R VisitValue(const Value& n, Args... args) {
    CHECK(n.defined());
    static FType vtable = InitVTable();
    return vtable(n, this, std::forward<Args>(args)...);
  }
  // Functions that can be overriden by subclass
  virtual R VisitValue_(const TensorValueObj* op,
                        Args... args) VALUE_FUNCTOR_DEFAULT;
  virtual R VisitValue_(const TensorTypeValueObj* op,
                        Args... args) VALUE_FUNCTOR_DEFAULT;
  virtual R VisitValue_(const TupleValueObj* op,
                       Args... args) VALUE_FUNCTOR_DEFAULT;
  virtual R VisitValue_(const RefValueObj* op,
                       Args... args) VALUE_FUNCTOR_DEFAULT;
  virtual R VisitValue_(const OpValueObj* op,
                       Args... args) VALUE_FUNCTOR_DEFAULT;
  virtual R VisitValue_(const OpaqueValueObj* op, Args... args) VALUE_FUNCTOR_DEFAULT;
  virtual R VisitValue_(const IntValueObj* op, Args... args) VALUE_FUNCTOR_DEFAULT;
  virtual R VisitValue_(const FloatValueObj* op,
                       Args... args) VALUE_FUNCTOR_DEFAULT;
  virtual R VisitValue_(const BoolValueObj* op,
                       Args... args) VALUE_FUNCTOR_DEFAULT;
  virtual R VisitValue_(const StringValueObj* op, Args... args) VALUE_FUNCTOR_DEFAULT;
  virtual R VisitValue_(const NoGradValueObj* op, Args... args) VALUE_FUNCTOR_DEFAULT;
  virtual R VisitValueDefault_(const ir::Object* op, Args...) {
    LOG(FATAL) << "Do not have a default for " << op->GetTypeKey();
    throw;
  }

 private:
  // initialize the vtable.
  static FType InitVTable() {
    FType vtable;
    // Set dispatch
    VALUE_FUNCTOR_DISPATCH(TensorValueObj);
    VALUE_FUNCTOR_DISPATCH(TensorTypeValueObj);
    VALUE_FUNCTOR_DISPATCH(TupleValueObj);
    VALUE_FUNCTOR_DISPATCH(RefValueObj);
    VALUE_FUNCTOR_DISPATCH(OpValueObj);
    VALUE_FUNCTOR_DISPATCH(OpaqueValueObj);
    VALUE_FUNCTOR_DISPATCH(IntValueObj);
    VALUE_FUNCTOR_DISPATCH(FloatValueObj);
    VALUE_FUNCTOR_DISPATCH(BoolValueObj);
    VALUE_FUNCTOR_DISPATCH(StringValueObj);
    VALUE_FUNCTOR_DISPATCH(NoGradValueObj);
    return vtable;
  }
};

}  // namespace value
}  // namespace mnm
