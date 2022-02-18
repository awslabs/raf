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
 * \file dialect.h
 * \brief Dialect and dialect operator interface
 */
#pragma once
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "./ir.h"
#include "./registry.h"

namespace mnm {
namespace op {

class DialectPreferenceObj : public Object {
 public:
  Array<String> preferred_dialects;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("preferred_dialects", &preferred_dialects);
  }

  static constexpr const char* _type_key = "mnm.op.DialectPreference";
  MNM_FINAL_OBJECT(DialectPreferenceObj, Object);
};

class DialectPreference : public ObjectRef {
 public:
  DialectPreference(Array<String> dialects);
  /*!
   * \brief Push a new dialect preference context onto the thread local stack.
   *  The DialectPreference on top of the stack is used to determine which
   *  dialect op to use when dispatching a base op.
   */
  void EnterWithScope();
  /*!
   * \brief Pop a dialect preference context off the thread local context stack,
   *  restoring the previous dialect preference as the current context.
   */
  void ExitWithScope();
  /*!
   * \brief Get the current dialect preference context from thread local storage.
   * \return The dialect preference that is the current context.
   */
  static const DialectPreference* Current();

  friend class tvm::With<DialectPreference>;
  MNM_OBJECT_REF(DialectPreference, ObjectRef, DialectPreferenceObj);
};

/*! \brief The dialect registry. */
class Dialect {
  using TRegistry = ::dmlc::Registry<Dialect>;

 public:
  Dialect() = default;
  /*! \brief Set the name of the dialect. */
  Dialect& set_name(std::string&& name);
  /*! \brief Set the device type to enable for this dialect. */
  Dialect& set_enable(DevType device_type);
  /*!
   * \brief Check if the dialect is enabled for the device type.
   * \param device_type The device type.
   * \return Whether the dialect is enabled. */
  bool is_enabled(DevType device_type) const;

  /*! \brief Get the registry. */
  static TRegistry* Registry();
  /*!
   * \brief Get the dialect given the name.
   * \param dialect_name The dialect name.
   * \return The dialect.
   */
  static const Dialect* Get(const std::string& dialect_name);
  /*!
   * \brief Check if a dialect is enabled on a given device type.
   * \param dialect The dialect name.
   * \param device_type The device type.
   * \return Whether the dialect is enabled.
   */
  static bool IsEnabled(const std::string& dialect, DevType device_type);
  /*!
   * \brief Get all enabled dialects given a device type.
   * \param device_type The device type.
   * \return A list of enabled dialects.
   */
  static std::vector<std::string> GetEnabledDialects(DevType device_type);

  /*! \brief The dialect name. */
  std::string name;

 private:
  /*! \brief The list of enabled devices. */
  std::vector<DevType> enable_devices_;
};

/*! \brief The dialect op registry for base ops. */
class OpDialect {
  /*! \brief Dialect op registry entry. */
  struct DialectOpEntry {
    /*! \brief The dialect name. */
    std::string dialect;
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
   * \param dialect_name The dialect name, e.g., "cudnn".
   * \param dialect_op The dialect op name, e.g., "mnm.op.cudnn.conv2d".
   * \param plevel The priority level.
   * \return The OpDialect itself.
   */
  OpDialect& add_dialect(const std::string& dialect_name, const std::string& dialect_op,
                         int plevel = 10);

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
                         const std::vector<std::string>& skip_dialects = {});
  /*!
   * \brief Lower a base op to the specified dialect.
   * \param base_op The base op.
   * \param device_type The device type.
   * \param dialect The dialect name.
   * \return The dialect op if it's registered; otherwise an undefined op.
   */
  static ir::Op Lower(const ir::Op& base_op, const std::string& dialect);

  /*! \brief The name of base op. */
  std::string name;

 private:
  /*! \brief The dialect ops registered to the base op. */
  TDialectList dialect_ops_;
  /*! \brief Mutex for dialect_ops_. */
  std::mutex dialect_ops_mu_;
};

/*!
 * \brief The dialect dispatch pattern data structure and registry.
 */
class DialectFusePattern {
 public:
  using PatternList = std::list<DialectFusePattern>;
  /*! \brief Pattern to dispatch base op(s) to dialect op(s). */
  ir::DFPattern pattern;
  /*! \brief The dialect name. */
  std::string dialect;
  /*! \brief The priority level. */
  int plevel;
  /*! \brief The pattern name. */
  std::string name;

  /*! \brief Get all dilaect fuse patterns. */
  static PatternList* Get();
  /*!
   * \brief Add a new dialect fuse pattern.
   * \param pattern The fuse pattern to match.
   * \param dialect The dialect name.
   * \param plevel The priority level.
   * \param name The pattern name.
   */
  static void AddPattern(const ir::DFPattern& pattern, const std::string& dialect, int plevel,
                         const std::string& name = "");
};

/*!
 * \brief Check if an op is a dialect op.
 * \param op The operator.
 * \return Whether it's a dialect op.
 */
bool IsDialectOp(const ir::Op& op);
/*!
 * \brief Get the dialect name given an op.
 * \param op The operator.
 * \return The dialect name. Return empty string if it's a base op.
 */
std::string GetDialect(const ir::Op& op);
/*!
 * \brief Get the base op given a dialect op.
 * \param dialect_op The dialect operator.
 * \return The corresponding base operator.
 */
ir::Op GetBaseOp(const ir::Op& dialect_op);

}  // namespace op
}  // namespace mnm

#define _MNM_DIALECT_DEF static DMLC_ATTRIBUTE_UNUSED ::mnm::op::Dialect& __make_##Dialect

#define _MNM_OP_DIALECT_DEF static DMLC_ATTRIBUTE_UNUSED ::mnm::op::OpDialect& __make_##OpDialect

#define _MNM_STRINGIZE(S) #S

#define MNM_BASE_OP_NAME(NAME) _MNM_STRINGIZE(mnm.op.NAME)

#define MNM_DIALECT_OP_NAME(DIALECT, NAME) _MNM_STRINGIZE(mnm.op.DIALECT.NAME)

#define MNM_REGISTER_DIALECT(DIALECT_NAME)         \
  DMLC_STR_CONCAT(_MNM_DIALECT_DEF, __COUNTER__) = \
      ::mnm::op::Dialect::Registry()->__REGISTER_OR_GET__(DIALECT_NAME).set_name(DIALECT_NAME)

#define MNM_REGISTER_DIALECT_OP(DIALECT, OP, PLEVEL)                        \
  DMLC_STR_CONCAT(_MNM_OP_DIALECT_DEF, __COUNTER__) =                       \
      ::mnm::op::OpDialect::Registry()                                      \
          ->__REGISTER_OR_GET__(MNM_BASE_OP_NAME(OP))                       \
          .set_name(MNM_BASE_OP_NAME(OP))                                   \
          .add_dialect(#DIALECT, MNM_DIALECT_OP_NAME(DIALECT, OP), PLEVEL); \
  RELAY_REGISTER_OP(MNM_DIALECT_OP_NAME(DIALECT, OP))                       \
      .set_attr<::mnm::op::TMNMDialect>("TMNMDialect", #DIALECT)            \
      .set_attr<::mnm::op::TMNMBaseOp>("TMNMBaseOp", MNM_BASE_OP_NAME(OP))
