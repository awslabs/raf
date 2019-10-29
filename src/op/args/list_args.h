#pragma once
#include <mnm/op.h>

namespace mnm {
namespace op {
namespace args {

class ListArgs : public ir::AttrsNode<ListArgs> {
 public:
  ir::Array<value::Value> args;

  MNM_OP_SCHEMA(ListArgs, "mnm.args.list") {
    this->args = args;
  }
};

}  // namespace args
}  // namespace op
}  // namespace mnm
