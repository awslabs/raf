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
 * \file serialization.h
 * \brief serialize & deserialize mnm extended node system.
 */
#pragma once
#include <tvm/node/structural_hash.h>
#include <tvm/runtime/object.h>
#include <string>
#include "./ir.h"
#include "./value.h"

namespace mnm {
namespace ir {
namespace serialization {

/**
 * Constant node for serialization, provides separate _type_key
 * to distinguish from relay.ConstantNode.
 */
class ConstantNode : public ir::ConstantNode {
 public:
  static constexpr const char* _type_key = "mnm.ir.serialization.Constant";
  MNM_FINAL_OBJECT(ConstantNode, ir::ConstantNode);
};

/*!
 * \brief Save as json string. Extended IR is converted before serialization.
 * \param node node registered in tvm node system.
 * \return serialized JSON string
 */
std::string SaveJSON(const ir::ObjectRef& node);

/*!
 * \brief Serialize value into byte stream.
 * \param strm DMLC stream.
 * \param value The value to be serialized.
 */
void SerializeValue(dmlc::Stream* strm, const value::Value& value);
/*!
 * \brief DeSerialize the value from the byte stream.
 * \param strm DMLC stream.
 * \return The value.
 */
value::Value DeserializeValue(dmlc::Stream* strm);

}  // namespace serialization
}  // namespace ir
}  // namespace mnm
