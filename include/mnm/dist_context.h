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
  int scheduling_param = 0;
  int iteration = 0;
  int root_rank = 0;
  int rank = 0;
  int size = 0;
  int local_rank = 0;
  int local_size = 0;
  bool enable_data_parallel = false;
  int zero_opt_level = 0;
  int auto_dp_profiling_start_iter = 2;
  int auto_dp_profiling_end_iter = 4;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("root_rank", &root_rank);
    v->Visit("rank", &rank);
    v->Visit("size", &size);
    v->Visit("local_rank", &local_rank);
    v->Visit("local_size", &local_size);
    v->Visit("enable_data_parallel", &enable_data_parallel);
    v->Visit("zero_opt_level", &zero_opt_level);
    v->Visit("auto_dp_profiling_start_iter", &auto_dp_profiling_start_iter);
    v->Visit("auto_dp_profiling_end_iter", &auto_dp_profiling_end_iter);
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
