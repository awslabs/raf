/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/distributed/dist_config.cc
 * \brief Config of Distributed Settings.
 */
#include "raf/registry.h"
#include "raf/communicator.h"
#include "raf/dist_config.h"

namespace raf {
namespace distributed {

using communicator::Communicator;
using communicator::CommunicatorPool;

DistConfig DistConfig::make() {
  ir::ObjectPtr<DistConfigObj> n = ir::make_object<DistConfigObj>();
  return DistConfig(n);
}

DistConfig DistConfig::Global() {
  static DistConfig inst = DistConfig::make();
  return inst;
}

void EnableDataParallel(bool enable) {
  DistConfig::Global()->enable_data_parallel = enable;
}

void ZeroOpt(int opt_level) {
  DistConfig::Global()->zero_opt_level = opt_level;
}

void EnableAutoDPProfiling(bool enable_auto_dp_profiling) {
  DistConfig::Global()->enable_auto_dp_profiling = enable_auto_dp_profiling;
}

void AutoDPProfilingStartIter(int auto_dp_profiling_start_iter) {
  DistConfig::Global()->auto_dp_profiling_start_iter = auto_dp_profiling_start_iter;
}

void AutoDPProfilingEndIter(int auto_dp_profiling_end_iter) {
  DistConfig::Global()->auto_dp_profiling_end_iter = auto_dp_profiling_end_iter;
}

RAF_REGISTER_GLOBAL("raf.distributed.GlobalDistConfig").set_body_typed(DistConfig::Global);
RAF_REGISTER_GLOBAL("raf.distributed.EnableDataParallel").set_body_typed(EnableDataParallel);
RAF_REGISTER_GLOBAL("raf.distributed.ZeroOpt").set_body_typed(ZeroOpt);
RAF_REGISTER_GLOBAL("raf.distributed.EnableAutoDPProfiling").set_body_typed(EnableAutoDPProfiling);
RAF_REGISTER_GLOBAL("raf.distributed.AutoDPProfilingStartIter")
    .set_body_typed(AutoDPProfilingStartIter);
RAF_REGISTER_GLOBAL("raf.distributed.AutoDPProfilingEndIter")
    .set_body_typed(AutoDPProfilingEndIter);

RAF_REGISTER_OBJECT_REFLECT(DistConfigObj);

}  // namespace distributed
}  // namespace raf
