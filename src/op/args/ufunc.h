#pragma once
#include "./utils.h"
namespace mnm {
namespace op {
namespace args {
class BinaryArgs : public ir::AttrsNode<BinaryArgs> {
 public:
  value::Value x1;
  value::Value x2;
  MNM_OP_SCHEMA(BinaryArgs, "mnm.args.binary") {
    MNM_ARG_REQUIRED(0, ir::Downcast<value::Value>, x1);
    MNM_ARG_REQUIRED(1, ir::Downcast<value::Value>, x2);
  }
};
class BinaryDxArgs : public ir::AttrsNode<BinaryDxArgs> {
 public:
  value::Value x1;
  value::Value x2;
  value::TensorValue y;
  value::TensorValue dy;
  MNM_OP_SCHEMA(BinaryDxArgs, "mnm.args.binary_dx") {
    MNM_ARG_REQUIRED(0, ir::Downcast<value::Value>, x1);
    MNM_ARG_REQUIRED(1, ir::Downcast<value::Value>, x2);
    MNM_ARG_REQUIRED(2, ir::Downcast<value::TensorValue>, y);
    MNM_ARG_REQUIRED(3, ir::Downcast<value::TensorValue>, dy);
  }
};
class BinaryUfuncArgs : public ir::AttrsNode<BinaryUfuncArgs> {
 public:
  value::Value x1;
  value::Value x2;
  value::Value out{nullptr};
  value::Value where{nullptr};
  MNM_OP_SCHEMA(BinaryUfuncArgs, "mnm.args.binary_ufunc") {
    MNM_ARG_REQUIRED(0, ir::Downcast<value::Value>, x1);
    MNM_ARG_REQUIRED(1, ir::Downcast<value::Value>, x2);
    MNM_ARG_OPTIONAL(2, ir::Downcast<value::Value>, out);
    MNM_ARG_OPTIONAL(3, ir::Downcast<value::Value>, where);
  }
};
class TernaryArgs : public ir::AttrsNode<TernaryArgs> {
 public:
  value::Value x1;
  value::Value x2;
  value::Value x3;
  MNM_OP_SCHEMA(TernaryArgs, "mnm.args.ternary") {
    MNM_ARG_REQUIRED(0, ir::Downcast<value::Value>, x1);
    MNM_ARG_REQUIRED(1, ir::Downcast<value::Value>, x2);
    MNM_ARG_REQUIRED(2, ir::Downcast<value::Value>, x3);
  }
};
class TernaryDxArgs : public ir::AttrsNode<TernaryDxArgs> {
 public:
  value::Value x1;
  value::Value x2;
  value::Value x3;
  value::TensorValue y;
  value::TensorValue dy;
  MNM_OP_SCHEMA(TernaryDxArgs, "mnm.args.ternary_dx") {
    MNM_ARG_REQUIRED(0, ir::Downcast<value::Value>, x1);
    MNM_ARG_REQUIRED(1, ir::Downcast<value::Value>, x2);
    MNM_ARG_REQUIRED(2, ir::Downcast<value::Value>, x3);
    MNM_ARG_REQUIRED(3, ir::Downcast<value::TensorValue>, y);
    MNM_ARG_REQUIRED(4, ir::Downcast<value::TensorValue>, dy);
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
    MNM_ARG_REQUIRED(0, ir::Downcast<value::Value>, x1);
    MNM_ARG_REQUIRED(1, ir::Downcast<value::Value>, x2);
    MNM_ARG_REQUIRED(2, ir::Downcast<value::Value>, x3);
    MNM_ARG_OPTIONAL(3, ir::Downcast<value::Value>, out);
    MNM_ARG_OPTIONAL(4, ir::Downcast<value::Value>, where);
  }
};
class UnaryArgs : public ir::AttrsNode<UnaryArgs> {
 public:
  value::Value x;
  MNM_OP_SCHEMA(UnaryArgs, "mnm.args.unary") {
    MNM_ARG_REQUIRED(0, ir::Downcast<value::Value>, x);
  }
};
class UnaryDxArgs : public ir::AttrsNode<UnaryDxArgs> {
 public:
  value::Value x;
  value::TensorValue y;
  value::TensorValue dy;
  MNM_OP_SCHEMA(UnaryDxArgs, "mnm.args.unary_dx") {
    MNM_ARG_REQUIRED(0, ir::Downcast<value::Value>, x);
    MNM_ARG_REQUIRED(1, ir::Downcast<value::TensorValue>, y);
    MNM_ARG_REQUIRED(2, ir::Downcast<value::TensorValue>, dy);
  }
};
class UnaryUfuncArgs : public ir::AttrsNode<UnaryUfuncArgs> {
 public:
  value::Value x;
  value::Value out{nullptr};
  value::Value where{nullptr};
  MNM_OP_SCHEMA(UnaryUfuncArgs, "mnm.args.unary_ufunc") {
    MNM_ARG_REQUIRED(0, ir::Downcast<value::Value>, x);
    MNM_ARG_OPTIONAL(1, ir::Downcast<value::Value>, out);
    MNM_ARG_OPTIONAL(2, ir::Downcast<value::Value>, where);
  }
};
}  // namespace args
}  // namespace op
}  // namespace mnm
