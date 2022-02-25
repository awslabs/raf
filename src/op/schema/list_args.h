/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/schema/list_args.h
 * \brief A list of arguments.
 */
#pragma once
#include <vector>
#include <string>
#include "raf/op.h"

namespace raf {
namespace op {
namespace schema {

class ListArgs : public ir::AttrsNode<ListArgs> {
 public:
  ir::Array<value::Value> args;
  RAF_OP_SCHEMA(ListArgs, "raf.args.list");
};

}  // namespace schema
}  // namespace op
}  // namespace raf
