/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file serialization.h
 * \brief serialize & deserialize raf extended node system.
 */
#pragma once
#include <tvm/node/structural_hash.h>
#include <tvm/runtime/object.h>
#include <string>
#include "./ir_ext.h"
#include "./value.h"

namespace raf {
namespace ir {
namespace serialization {

/**
 * Constant node for serialization, provides separate _type_key
 * to distinguish from relay.ConstantNode.
 */
class ConstantNode : public ir::ConstantNode {
 public:
  static constexpr const char* _type_key = "raf.ir.serialization.Constant";
  RAF_FINAL_OBJECT_NOCHECK(ConstantNode, ir::ConstantNode);
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
}  // namespace raf
