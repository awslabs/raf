#pragma once
#include <mnm/op.h>

namespace mnm {
namespace op {
namespace args {

class UnaryUfuncArgs : public ir::AttrsNode<UnaryUfuncArgs> {
 public:
  value::Value x;
  value::Value out{nullptr};
  value::Value where{nullptr};

  MNM_OP_SCHEMA(UnaryUfuncArgs, "mnm.args.unary_ufunc") {
    MNM_ARG_REQUIRED(0, value::Value, x);
    MNM_ARG_OPTIONAL(1, value::Value, out);
    MNM_ARG_OPTIONAL(2, value::Value, where);
  }
};

class BinaryUfuncArgs : public ir::AttrsNode<BinaryUfuncArgs> {
 public:
  value::Value x1;
  value::Value x2;
  value::Value out{nullptr};
  value::Value where{nullptr};

  MNM_OP_SCHEMA(BinaryUfuncArgs, "mnm.args.binary_ufunc") {
    MNM_ARG_REQUIRED(0, value::Value, x1);
    MNM_ARG_REQUIRED(1, value::Value, x2);
    MNM_ARG_OPTIONAL(2, value::Value, out);
    MNM_ARG_OPTIONAL(3, value::Value, where);
  }
};

class TernaryUfuncArgs : public ir::AttrsNode<TernaryUfuncArgs> {
 public:
  value::Value x1;
  value::Value x2;
  value::Value x3;
  value::Value out{nullptr};
  value::Value where{nullptr};

  MNM_OP_SCHEMA(TernaryUfuncArgs, "mnm.args.ternary_ufunc") {
    MNM_ARG_REQUIRED(0, value::Value, x1);
    MNM_ARG_REQUIRED(1, value::Value, x2);
    MNM_ARG_REQUIRED(2, value::Value, x3);
    MNM_ARG_OPTIONAL(3, value::Value, out);
    MNM_ARG_OPTIONAL(4, value::Value, where);
  }
};

class UnaryArgs : public ir::AttrsNode<UnaryArgs> {
 public:
  value::Value x;

  MNM_OP_SCHEMA(UnaryArgs, "mnm.args.unary") {
    MNM_ARG_REQUIRED(0, value::Value, x);
  }
};

class UnaryDxArgs : public ir::AttrsNode<UnaryArgs> {
 public:
  value::Value x;
  value::Value y;
  value::Value dy;

  MNM_OP_SCHEMA(UnaryDxArgs, "mnm.args.unary_dx") {
    MNM_ARG_REQUIRED(0, value::Value, x);
    MNM_ARG_REQUIRED(1, value::Value, y);
    MNM_ARG_REQUIRED(2, value::Value, dy);
  }
};

class BinaryArgs : public ir::AttrsNode<BinaryArgs> {
 public:
  value::Value x1;
  value::Value x2;

  MNM_OP_SCHEMA(BinaryArgs, "mnm.args.binary") {
    MNM_ARG_REQUIRED(0, value::Value, x1);
    MNM_ARG_REQUIRED(1, value::Value, x2);
  }
};

class TernaryArgs : public ir::AttrsNode<TernaryArgs> {
 public:
  value::Value x1;
  value::Value x2;
  value::Value x3;

  MNM_OP_SCHEMA(TernaryArgs, "mnm.args.ternary") {
    MNM_ARG_REQUIRED(0, value::Value, x1);
    MNM_ARG_REQUIRED(1, value::Value, x2);
    MNM_ARG_REQUIRED(2, value::Value, x3);
  }
};

}  // namespace args
}  // namespace op
}  // namespace mnm
