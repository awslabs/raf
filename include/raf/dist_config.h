/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file dist_config.h
 * \brief Config of Distributed Settings.
 */
#pragma once
#include "./ir.h"
#include "./communicator.h"

namespace raf {
namespace distributed {

class DistConfigObj : public ir::Object {
 public:
  int scheduling_param = 0;
  int iteration = 0;
  bool enable_data_parallel = false;
  int zero_opt_level = 0;
  bool enable_auto_dp_profiling = false;
  int auto_dp_profiling_start_iter = 2;
  int auto_dp_profiling_end_iter = 4;
  int64_t group_bucket_size = 5000000000;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("enable_data_parallel", &enable_data_parallel);
    v->Visit("zero_opt_level", &zero_opt_level);
    v->Visit("enable_auto_dp_profiling", &enable_auto_dp_profiling);
    v->Visit("auto_dp_profiling_start_iter", &auto_dp_profiling_start_iter);
    v->Visit("auto_dp_profiling_end_iter", &auto_dp_profiling_end_iter);
    v->Visit("group_bucket_size", &group_bucket_size);
  }

 public:
  static constexpr const char* _type_key = "raf.distributed.DistConfig";
  RAF_FINAL_OBJECT(DistConfigObj, ir::Object);
};

class DistConfig : public ir::ObjectRef {
 public:
  static DistConfig make();
  static DistConfig Global();
  RAF_MUTABLE_OBJECT_REF(DistConfig, ir::ObjectRef, DistConfigObj);
};

}  // namespace distributed
}  // namespace raf
