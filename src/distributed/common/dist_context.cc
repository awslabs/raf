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
 * \file src/distributed/context.cc
 * \brief Context of Distributed Settings.
 */
#include "mnm/registry.h"
#include "mnm/dist_context.h"

namespace mnm {
namespace distributed {

using communicator::Communicator;
using communicator::CommunicatorPool;

DistContext DistContext::make() {
  /* Legacy Support */
  ir::ObjectPtr<DistContextObj> n = ir::make_object<DistContextObj>();
  Communicator comm = Communicator::Get();
  n->root_rank = comm->root_rank;
  n->rank = comm->rank;
  n->size = comm->size;
  n->local_rank = comm->local_rank;
  n->local_size = comm->local_size;

  return DistContext(n);
}

DistContext DistContext::Global() {
  static DistContext inst = DistContext::make();
  return inst;
}

void EnableDataParallel(bool enable) {
  DistContext::Global()->enable_data_parallel = enable;
}

void ZeroOpt(int opt_level) {
  DistContext::Global()->zero_opt_level = opt_level;
}

void SetGlobalRank(int rank) {
  DistContext::Global()->rank = rank;
}

void SetGlobalSize(int size) {
  DistContext::Global()->size = size;
}

void AutoDPProfilingStartIter(int auto_dp_profiling_start_iter) {
  DistContext::Global()->auto_dp_profiling_start_iter = auto_dp_profiling_start_iter;
}

void AutoDPProfilingEndIter(int auto_dp_profiling_end_iter) {
  DistContext::Global()->auto_dp_profiling_end_iter = auto_dp_profiling_end_iter;
}

MNM_REGISTER_GLOBAL("mnm.distributed._make.DistContext").set_body_typed(DistContext::make);
MNM_REGISTER_GLOBAL("mnm.distributed.Global").set_body_typed(DistContext::Global);
MNM_REGISTER_GLOBAL("mnm.distributed.EnableDataParallel").set_body_typed(EnableDataParallel);
MNM_REGISTER_GLOBAL("mnm.distributed.ZeroOpt").set_body_typed(ZeroOpt);
MNM_REGISTER_GLOBAL("mnm.distributed.SetGlobalRank").set_body_typed(SetGlobalRank);
MNM_REGISTER_GLOBAL("mnm.distributed.SetGlobalSize").set_body_typed(SetGlobalSize);
MNM_REGISTER_GLOBAL("mnm.distributed.AutoDPProfilingStartIter")
    .set_body_typed(AutoDPProfilingStartIter);
MNM_REGISTER_GLOBAL("mnm.distributed.AutoDPProfilingEndIter")
    .set_body_typed(AutoDPProfilingEndIter);

MNM_REGISTER_OBJECT_REFLECT(DistContextObj);

}  // namespace distributed
}  // namespace mnm
