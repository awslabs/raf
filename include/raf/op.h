/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file op.h
 * \brief Operator interface
 */
#pragma once
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "./device.h"
#include "./dialect.h"
#include "./ir.h"
#include "./registry.h"
#include "./value.h"
#include "./memory_pool.h"

namespace raf {
namespace executor {
class Executor;
}  // namespace executor
namespace requests {
class Requests;
}  // namespace requests
}  // namespace raf

namespace raf {
namespace op {

class CallValuesNode : public ir::Object {
 public:
  mutable value::Value callee;
  mutable ir::Attrs args;
  mutable value::Value out;
  mutable Device device;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("callee", &callee);
    v->Visit("args", &args);
    v->Visit("out", &out);
    v->Visit("device", &device);
  }
  static constexpr const uint32_t _type_index = ir::TypeIndex::kDynamic;
  static constexpr const char* _type_key = "raf.op.CallValues";
  RAF_FINAL_OBJECT(CallValuesNode, ir::Object);
};

class CallValues : public ir::ObjectRef {
 public:
  static CallValues make(value::Value callee = {}, ir::Attrs args = {});
  RAF_OBJECT_REF(CallValues, ir::ObjectRef, CallValuesNode);
};

class OpEnv {
  class Impl;
  std::shared_ptr<Impl> impl;

 public:
  OpEnv();
  virtual ~OpEnv();
  /*! \brief Return the name of the OpEnv. */
  virtual std::string name() const = 0;
  /*!
   * \brief Execute the OpEnv with call values.
   * \param call The call values.
   */
  virtual void Execute(const CallValues& call) = 0;
  /*!
   * \brief Execute the OpEnv with a list of inputs and output value.
   * \param inputs The vector of input values.
   * \param output The output value.
   */
  virtual void Execute(const std::vector<value::Value>& inputs, value::Value output) = 0;

  /*! \brief Whether this OpEnv is valid. */
  std::vector<std::string> error_msgs;
  bool HasError() {
    return !error_msgs.empty();
  }

  void RequestWorkspace(void** dest, const Device& device, int64_t nbytes);
  void RequestStream(void** dest, const Device& device, int tag_idx);
  void RequestDistributed(void** dest, const std::string& name, const value::Value rank_list);

  void BindExecutor(executor::Executor* executor);
  std::shared_ptr<requests::Requests> GetRequests() const;
  /*! \brief Data input indices in the argument list. This is used by VM executor. */
  std::vector<int> arg_indices;

  /*!
   * \brief Set the stream to launch the kernels for all enabled backends
   * \param device The device of the stream.
   * \param stream The stream to launch the operators on.
   */
  static void SetStreamForAllBackends(Device device, void* stream);
};

using OpEnvPtr = std::shared_ptr<OpEnv>;

/*!
 * \brief Registry to make an OpEnv for an operator.
 */
class OpEnvMaker {
  using TRegistry = ::dmlc::Registry<OpEnvMaker>;

 public:
  using FMakeOpEnv = std::function<OpEnv*(const CallValues& call)>;

  OpEnvMaker() = default;
  OpEnvMaker(FMakeOpEnv f) : func_(f) {
  }
  /*! \brief Set the op name. */
  OpEnvMaker& set_name(const std::string& name);
  /*! \brief Set the OpEnv maker function. */
  OpEnvMaker& set_func(FMakeOpEnv f);
  /*! \brief Invoke the maker function. */
  OpEnv* operator()(const CallValues& call) const {
    CHECK(func_ != nullptr);
    return func_(call);
  }

  /*! \brief Get the registry. */
  static TRegistry* Registry();
  /*! \brief Get the OpEnvMaker given the operator name. */
  static const OpEnvMaker* Get(const std::string& op_name);
  /*!
   * \brief Make an OpEnv given the operator and call value.
   * \param op The operator.
   * \param call The call value.
   * \return The generated OpEnv.
   */
  static std::shared_ptr<OpEnv> Make(const std::string& op_name, const CallValues& call);

  /*! \brief The op name. */
  std::string name;

 private:
  /*! \brief The reigstered function. */
  FMakeOpEnv func_ = nullptr;
};

// Operator helper functions

/*! \brief Convert a list of values into ListArgs. */
ir::Attrs MakeListArgs(const ir::Array<value::Value>& values);
/*! \brief Retrieve the list of values from ListArgs. */
ir::Array<value::Value> GetListArgs(const ir::Attrs& attrs);
/*!
 * \brief Find an unallocated name for the given name.
 * \param name The given name
 * \return An unallocated name with a unique suffix attached
 */
std::string GetUniqueName(std::string name);
/*!
 * \brief Truncate the given name to fit in 80 characters
 * \param name The given name
 * \return The truncated name
 */
std::string TruncateName(std::string name);
/*!
 * \brief Get the operator attributes.
 *
 *   If `op` is a dialect op, the function will first check whether this attribute is registered
 *   to it. If not, it will try to retrieve the attribute from its base op.
 *
 * \param op The operator.
 * \param attr_name The attribute name.
 * \return The attribute.
 */
template <class T>
inline T GetOpAttr(const ir::Op& op, const std::string attr_name);
/*!
 * \brief Get the operator attributes with a default value.
 *
 *   Same as GetOpAttr, but returns the default value when the attribute is registered to
 *   neither the dialect op nor the base op.
 *
 * \param op The operator.
 * \param attr_name The attribute name.
 * \param default_value The default value.
 * \return The attribute or the default value.
 */
template <class T>
inline T GetOpAttrOrDefault(const ir::Op& op, const std::string attr_name, T default_value);
/*!
 * \brief Dispatch (fused or un-fused) ops to backend implementation.
 * \param call The call values.
 * \return The created OpEnv.
 */
std::shared_ptr<OpEnv> Dispatch(const CallValues& call);

/*!
 * \brief Create a dummy call_values from a call expression. The inputs and output of the call
 * values are dummy values created according to the inferred type of the call expression.
 * \param call The call expression.
 * \param device The target device.
 * \return The created dummy call_values.
 */
CallValues CreateDummyCallValues(Call call, Device device);

// Operator pattern
using tvm::relay::kBroadcast;
using tvm::relay::kCommReduce;
using tvm::relay::kElemWise;
using tvm::relay::kInjective;
using tvm::relay::kOpaque;
using tvm::relay::kOutEWiseFusable;
using tvm::relay::kTuple;
using tvm::relay::OpPatternKind;
using tvm::relay::TOpPattern;

/*! \brief indicate whether this operator has side effect. */
using TRAFSideEffect = bool;
/*! \brief indicate whether this operator is a collective communication op. */
using TRAFCollective = bool;
/*! \brief Map from input index to output index that the output share the memory with input. */
using TRAFInplaceUpdate = ir::Map<ir::Integer, ir::Integer>;
/*! \brief Indicate which dialect this dialect op belongs to. */
using TRAFDialect = std::string;
/*! \brief Indicate which base op this dialect op maps to. */
using TRAFBaseOp = std::string;

using FRAFDeclare = registry::TypedPackedFunc<void(const CallValues& call)>;

using FRAFSchema = registry::TypedPackedFunc<ir::Attrs(const ir::Array<value::Value>&)>;

/*!
 * \brief Generate the cast rule of an op given the current input expressions.
 * \param args An array of input expressions.
 * \param ret_type Return type of the current op.
 * \param amp_dtype The desired AMP dtype.
 * \return An array of type hints. The size is #arguments + 1 (output).
 */
using FRAFCastRule = registry::TypedPackedFunc<ir::Array<ir::Type>(
    const ir::Array<ir::Expr>&, const ir::Type&, const std::string&)>;

using FRAFAnnotateTarget =
    registry::TypedPackedFunc<bool(const ir::Attrs& attrs, const ir::Array<ir::Expr>& args)>;

using FRAFSchemaFieldIndex = registry::TypedPackedFunc<int(const std::string&)>;

using FPrimalGradient = registry::TypedPackedFunc<
    // returns: op's contribution to igrads
    ir::Array<ir::Expr>(
        // orig_call: a relay::Call which invokes this operator
        const ir::Expr& orig_call,
        // orig_args: Array of original args of the call node
        const ir::Array<ir::Expr> orig_args,
        // computed_output: (optional) to which var the op's output binds to
        const ir::Var& computed_output,
        // out_grad: ograds
        const ir::Expr& out_grad)>;

using FFusedPrimalGradient = registry::TypedPackedFunc<
    // returns: the updated igrads
    ir::Array<ir::Expr>(
        // orig_call: a relay::Call which invokes this operator
        const ir::Expr& orig_call,
        // computed_output: (optional) to which var the op's output binds to
        const ir::Var& computed_output,
        // out_grad: ograds
        const ir::Expr& out_grad,
        // in_grad: old igrads
        const ir::Array<ir::Expr>& in_grad)>;

using FRAFFromRelay =
    registry::TypedPackedFunc<ir::Expr(const ir::Attrs& attrs, const ir::Array<ir::Expr>& args,
                                       const ir::Map<ir::Var, ir::Expr>& val_map)>;
using FRAFMutationFromRelay = registry::TypedPackedFunc<ir::Array<ir::Array<ir::Expr>>(
    const ir::Var& var, const ir::Call& call)>;

// Implementation
template <class T>
inline std::pair<bool, T> _TryRetrieveAttr(const ir::Op& op, const std::string attr_name) {
  static auto fattr = ir::Op::GetAttrMap<T>(attr_name);
  if (fattr.count(op)) {
    return std::make_pair(true, fattr[op]);
  }
  if (IsDialectOp(op)) {
    auto base_op = GetBaseOp(op);
    if (fattr.count(base_op)) {
      return std::make_pair(true, fattr[base_op]);
    }
  }
  return std::make_pair(false, T());
}

template <class T>
inline T GetOpAttr(const ir::Op& op, const std::string attr_name) {
  auto optional_value = _TryRetrieveAttr<T>(op, attr_name);
  if (!optional_value.first) {
    LOG(FATAL) << "No attribute " << attr_name << " registered for " << op->name;
  }
  return optional_value.second;
}

template <class T>
inline T GetOpAttrOrDefault(const ir::Op& op, const std::string attr_name, T default_value) {
  auto optional_value = _TryRetrieveAttr<T>(op, attr_name);
  if (!optional_value.first) {
    return default_value;
  }
  return optional_value.second;
}

}  // namespace op
}  // namespace raf

#define _RAF_OP_DIALECT_DEF static DMLC_ATTRIBUTE_UNUSED ::raf::op::OpDialect& __make_##OpDialect

#define _RAF_OP_ENV_MAKER_DEF \
  static DMLC_ATTRIBUTE_UNUSED ::raf::op::OpEnvMaker& __make_##OpEnvMaker

#define _RAF_DIALECT_DEF static DMLC_ATTRIBUTE_UNUSED ::raf::op::Dialect& __make_##Dialect

#define _RAF_OP_DISPATCH_DEF static DMLC_ATTRIBUTE_UNUSED ::raf::op::OpDispatch& __make_##OpDispatch

#define _RAF_FUNC_DISPATCH_DEF \
  static DMLC_ATTRIBUTE_UNUSED ::raf::op::FusedOpDispatch& __make_##FusedOpDispatch

#define RAF_REGISTER_OP(OP_NAME) RELAY_REGISTER_OP(OP_NAME)

#define RAF_OP_DECLARE(OP_NAME, BODY) \
  RELAY_REGISTER_OP(OP_NAME).set_attr<::raf::op::FRAFDeclare>("FRAFDeclare", BODY)

#define RAF_OP_ENV_MAKER(OP_NAME, FOP_ENV_MAKER)                                                  \
  DMLC_STR_CONCAT(_RAF_OP_ENV_MAKER_DEF, __COUNTER__) =                                           \
      ::raf::op::OpEnvMaker::Registry()->__REGISTER_OR_GET__(OP_NAME).set_name(OP_NAME).set_func( \
          FOP_ENV_MAKER)

#define RAF_OP_SCHEMA(class_name, type_key)          \
  static constexpr const char* _type_key = type_key; \
  RAF_FINAL_OBJECT(class_name, ::tvm::BaseAttrsNode) \
  template <typename FVisit>                         \
  void _tvm_VisitAttrs(FVisit& _tvm_fvisit) {        \
  }

#define RAF_OP_GRAD(op_name, body) \
  RELAY_REGISTER_OP(op_name).set_attr<::raf::op::FPrimalGradient>("FPrimalGradient", body);

#define RAF_OP_FUSED_GRAD(op_name, body)                                                       \
  RELAY_REGISTER_OP(op_name).set_attr<::raf::op::FFusedPrimalGradient>("FFusedPrimalGradient", \
                                                                       body);
#define RAF_OP_GRAD_SKIP_INPUTS(op_name, body) \
  RELAY_REGISTER_OP(op_name).set_attr<std::string>("GradientInputSkip", body);
