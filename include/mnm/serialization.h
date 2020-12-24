/*!
 * Copyright (c) 2020 by Contributors
 * \file serialization.h
 * \brief serialize & deserialize mnm extended node system.
 */
#pragma once
#include <tvm/node/structural_hash.h>
#include <tvm/runtime/object.h>
#include <string>
#include "./ir.h"

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
std::string SaveJSON(const ir::Module& node);
std::string SaveJSON(const ir::Expr& node);
std::string SaveJSON(const ir::ObjectRef& node);

}  // namespace serialization
}  // namespace ir
}  // namespace mnm
