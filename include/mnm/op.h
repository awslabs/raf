/*!
 * Copyright (c) 2019 by Contributors
 * \file op.h
 * \brief Operator interface
 */
#pragma once
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "./device.h"
#include "./ir.h"
#include "./registry.h"
#include "./value.h"
#include "./memory_pool.h"

namespace mnm {
namespace executor {
class Executor;
}  // namespace executor
namespace requests {
class Requests;
}  // namespace requests
}  // namespace mnm

namespace mnm {
namespace op {

class CallValuesNode : public ir::Object {
 public:
  mutable value::Value callee;
  mutable ir::Attrs args;
  mutable value::Value out;
  mutable Device device;

 public:
  static constexpr const char* _type_key = "mnm.op.CallValues";
  MNM_FINAL_OBJECT(CallValuesNode, ir::Object);
};

class CallValues : public ir::ObjectRef {
 public:
  static CallValues make(value::Value callee = {}, ir::Attrs args = {});
  MNM_OBJECT_REF(CallValues, ir::ObjectRef, CallValuesNode);
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

  void RequestWorkspace(void** dest, const Device& device, int64_t nbytes);
  void RequestStream(void** dest, const Device& device, int tag_idx);
  void RequestDistributed(void** dest);

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
  /*! \brief Get the OpEnvMaker given an operator. */
  static const OpEnvMaker* Get(const ir::Op& op);
  /*!
   * \brief Make an OpEnv given the operator and call value.
   * \param op The operator.
   * \param call The call value.
   * \return The generated OpEnv.
   */
  static std::shared_ptr<OpEnv> Make(const ir::Op& op, const CallValues& call);

  /*! \brief The op name. */
  std::string name;

 private:
  /*! \brief The reigstered function. */
  FMakeOpEnv func_ = nullptr;
};

/*!
 * \brief The dialect registry for base ops.
 */
class OpDialect {
  /*! \brief Dialect op registry entry. */
  struct DialectOpEntry {
    /*! \brief The dialect name. Using "backend" is because we can reuse get_preferred_backends. */
    std::string backend;
    /*! \brief The name of dialect op. */
    std::string dialect_op;
    /*! \brief Priority level for this dialect op. */
    int plevel;
  };

  using TRegistry = ::dmlc::Registry<OpDialect>;
  using TDialectList = std::list<DialectOpEntry>;

 public:
  OpDialect() = default;
  /*! \brief Set the name of base op and return the OpDialect itself. */
  OpDialect& set_name(const std::string& name);
  /*!
   * \brief Register a dialect op to the base op.
   * \param device_type The device type.
   * \param dialect_name The dialect name, e.g., "cudnn".
   * \param dialect_op The dialect op name, e.g., "mnm.op.cudnn.conv2d".
   * \param plevel The priority level.
   * \return The OpDialect itself.
   */
  OpDialect& add_dialect(DevType device_type, const std::string& dialect_name,
                         const std::string& dialect_op, int plevel = 10);

  /*! \brief Get the registry. */
  static TRegistry* Registry();
  /*!
   * \brief Get the dialect dispatch list given a base op and device type.
   * \param op The base op.
   * \param device_type The device type.
   * \return The dialect dispatch list, ordered by the dialect plevel.
   */
  static TDialectList GetDispatchList(const ir::Op& op, DevType device_type);
  /*!
   * \brief Dispatch a base op to a dialect op.
   * \param base_op The base op.
   * \param device_type The device type.
   * \param skip_dialects The list of dialects to be skipped.
   * \return The dialect op.
   */
  static ir::Op Dispatch(const ir::Op& base_op, DevType device_type,
                         std::vector<std::string> skip_dialects = {});
  /*!
   * \brief Dispatch a base op to the specified dialect.
   * \param base_op The base op.
   * \param device_type The device type.
   * \param dialect The dialect name.
   * \return The dialect op if it's registered; otherwise an undefined op.
   */
  static ir::Op Dispatch(const ir::Op& base_op, DevType device_type, const std::string& dialect);

  /*! \brief The name of base op. */
  std::string name;
  /*! \brief The dialects registered to the base op. */
  registry::PerDevTypeStore<TDialectList> dialects;
};

/*!
 * \brief Singleton dispatcher for fused ops
 */
class FusedOpDispatch {
  struct FuncEnvMaker {
    int plevel;
    std::string backend;
    OpEnvMaker maker;
  };
  using TDispatchList = std::list<FuncEnvMaker>;

 private:
  FusedOpDispatch() = default;

 public:
  /*!
   * \brief add a dispatch for a specific backend
   * \param device_type the device type
   * \param backend_name the backend name
   * \param op_env_maker a function that converts a call value into the corresponding op env
   * \param plevel the backend priority. Backends with higher priority is preferred in dispatch.
   */
  FusedOpDispatch& add_dispatch(DevType device_type, const std::string& backend_name,
                                const OpEnvMaker::FMakeOpEnv& op_env_maker, int plevel = 10);

 public:
  /*! \brief get the FusedOpDispatch instance */
  static FusedOpDispatch* Get();
  /*! \brief get the list of all available backends on a specific device */
  static TDispatchList* Get(DevType device_type);
  /*! \brief dispatch call to some backend according to its device */
  static std::shared_ptr<OpEnv> Dispatch(const CallValues& call);

 public:
  /*! \brief the list of backends and their dispatch mathods available on different devices */
  registry::PerDevTypeStore<TDispatchList> dispatch;
};

/*! \brief Check if an op is a dialect op. */
bool IsDialectOp(const ir::Op& op);
/*! \brief Get the dialect name given an op. Return empty string if it's a base op. */
std::string GetDialect(const ir::Op& op);
/*! \brief Dispatch (fused or un-fused) ops to backend implementation */
std::shared_ptr<OpEnv> Dispatch(const CallValues& call);
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
using TMNMSideEffect = bool;

/*! \brief Map from input index to output index that the output share the memory with input. */
using TMNMInplaceUpdate = ir::Map<ir::Integer, ir::Integer>;
/*! \brief Indicate which dialect this dialect op belongs to. */
using TMNMDialect = std::string;
/*! \brief Indicate which base op this dialect op maps to. */
using TMNMBaseOp = std::string;

using FMNMDeclare = registry::TypedPackedFunc<void(const CallValues& call)>;

using FMNMSchema = registry::TypedPackedFunc<ir::Attrs(const ir::Array<value::Value>&)>;

/*!
 * \brief Generate the cast rule of an op given the current input expressions.
 * \param args An array of input expressions.
 * \param ret_type Return type of the current op.
 * \param amp_dtype The desired AMP dtype.
 * \return An array of type hints. The size is #arguments + 1 (output).
 */
using FMNMCastRule = registry::TypedPackedFunc<ir::Array<ir::Type>(
    const ir::Array<ir::Expr>&, const ir::Type&, const std::string&)>;

using FMNMAnnotateTarget =
    registry::TypedPackedFunc<bool(const ir::Attrs& attrs, const ir::Array<ir::Expr>& args)>;

using FMNMSchemaFieldIndex = registry::TypedPackedFunc<int(const std::string&)>;

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

using FMNMFromRelay =
    registry::TypedPackedFunc<ir::Expr(const ir::Attrs& attrs, const ir::Array<ir::Expr>& args,
                                       const ir::Map<ir::Var, ir::Expr>& val_map)>;
using FMNMMutationFromRelay = registry::TypedPackedFunc<ir::Array<ir::Array<ir::Expr>>(
    const ir::Var& var, const ir::Call& call)>;

}  // namespace op
}  // namespace mnm

#define _MNM_OP_DIALECT_DEF static DMLC_ATTRIBUTE_UNUSED ::mnm::op::OpDialect& __make_##OpDialect

#define _MNM_OP_ENV_MAKER_DEF \
  static DMLC_ATTRIBUTE_UNUSED ::mnm::op::OpEnvMaker& __make_##OpEnvMaker

#define _MNM_OP_DISPATCH_DEF static DMLC_ATTRIBUTE_UNUSED ::mnm::op::OpDispatch& __make_##OpDispatch

#define _MNM_FUNC_DISPATCH_DEF \
  static DMLC_ATTRIBUTE_UNUSED ::mnm::op::FusedOpDispatch& __make_##FusedOpDispatch

#define _MNM_STRINGIZE(S) #S

#define MNM_BASE_OP_NAME(NAME) _MNM_STRINGIZE(mnm.op.NAME)

#define MNM_DIALECT_OP_NAME(DIALECT, NAME) _MNM_STRINGIZE(mnm.op.DIALECT.NAME)

#define MNM_REGISTER_OP(OP_NAME) RELAY_REGISTER_OP(OP_NAME)

#define MNM_REGISTER_DIALECT_OP(DIALECT, OP)                     \
  RELAY_REGISTER_OP(MNM_DIALECT_OP_NAME(DIALECT, OP))            \
      .set_attr<::mnm::op::TMNMDialect>("TMNMDialect", #DIALECT) \
      .set_attr<::mnm::op::TMNMBaseOp>("TMNMBaseOp", MNM_BASE_OP_NAME(OP))

#define MNM_OP_DECLARE(OP_NAME, BODY) \
  RELAY_REGISTER_OP(OP_NAME).set_attr<::mnm::op::FMNMDeclare>("FMNMDeclare", BODY)

#define MNM_OP_ENV_MAKER(OP_NAME, FOP_ENV_MAKER)                                                  \
  DMLC_STR_CONCAT(_MNM_OP_ENV_MAKER_DEF, __COUNTER__) =                                           \
      ::mnm::op::OpEnvMaker::Registry()->__REGISTER_OR_GET__(OP_NAME).set_name(OP_NAME).set_func( \
          FOP_ENV_MAKER)

#define MNM_OP_DISPATCH_DIALECT_PLEVEL(OP, DIALECT, DEVICE_TYPE, PLEVEL) \
  DMLC_STR_CONCAT(_MNM_OP_DIALECT_DEF, __COUNTER__) =                    \
      ::mnm::op::OpDialect::Registry()                                   \
          ->__REGISTER_OR_GET__(MNM_BASE_OP_NAME(OP))                    \
          .set_name(MNM_BASE_OP_NAME(OP))                                \
          .add_dialect(DEVICE_TYPE, #DIALECT, MNM_DIALECT_OP_NAME(DIALECT, OP), PLEVEL)

#define MNM_FUNC_DISPATCH_PLEVEL(func_env_maker, device_type, backend_name, plevel)              \
  DMLC_STR_CONCAT(_MNM_FUNC_DISPATCH_DEF, __COUNTER__) =                                         \
      ::mnm::op::FusedOpDispatch::Get()->add_dispatch(device_type, backend_name, func_env_maker, \
                                                      plevel)

#define MNM_FUNC_DISPATCH(func_env_maker, device_type, backend_name) \
  MNM_FUNC_DISPATCH_PLEVEL(func_env_maker, device_type, backend_name, 10)

#define MNM_OP_SCHEMA(class_name, type_key)          \
  static constexpr const char* _type_key = type_key; \
  MNM_FINAL_OBJECT(class_name, ::tvm::BaseAttrsNode) \
  template <typename FVisit>                         \
  void __VisitAttrs__(FVisit& __fvisit__) {          \
  }

#define MNM_OP_GRAD(op_name, body) \
  RELAY_REGISTER_OP(op_name).set_attr<::mnm::op::FPrimalGradient>("FPrimalGradient", body);

#define MNM_OP_FUSED_GRAD(op_name, body)                                                       \
  RELAY_REGISTER_OP(op_name).set_attr<::mnm::op::FFusedPrimalGradient>("FFusedPrimalGradient", \
                                                                       body);
#define MNM_OP_GRAD_SKIP_INPUTS(op_name, body) \
  RELAY_REGISTER_OP(op_name).set_attr<std::string>("GradientInputSkip", body);
