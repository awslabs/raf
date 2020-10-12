/*!
 * Copyright (c) 2020 by Contributors
 * \file dist_context.h
 * \brief Context of Distributed Settings.
 */
#pragma once
#include "./ir.h"
#include "./communicator.h"

namespace mnm {
namespace distributed {

class DistContext;

class DistContextObj : public ir::Object {
 public:
  bool enable_data_parallel = false;
  int root_rank = 0;
  int rank = 0;
  int size = 0;
  int local_rank = 0;
  int local_size = 0;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("enable_data_parallel", &enable_data_parallel);
    v->Visit("root_rank", &root_rank);
    v->Visit("rank", &rank);
    v->Visit("size", &size);
    v->Visit("local_rank", &local_rank);
    v->Visit("local_size", &local_size);
  }

 public:
  static constexpr const char* _type_key = "mnm.distributed.DistContext";
  MNM_FINAL_OBJECT(DistContextObj, ir::Object);

  friend class DistContext;
};

class DistContext : public ir::ObjectRef {
 public:
  static DistContext make();
  static DistContext Global();
  MNM_OBJECT_REF(DistContext, ir::ObjectRef, DistContextObj);
};

}  // namespace distributed
}  // namespace mnm
