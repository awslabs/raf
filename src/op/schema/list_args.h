/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/schema/list_args.h
 * \brief A list of arguments.
 */
#pragma once
#include <vector>
#include <string>
#include "mnm/op.h"

namespace mnm {
namespace op {
namespace schema {

class ListArgs : public ir::AttrsNode<ListArgs> {
 public:
  ir::Array<value::Value> args;
  MNM_OP_SCHEMA(ListArgs, "mnm.args.list");
};

}  // namespace schema
}  // namespace op
}  // namespace mnm
