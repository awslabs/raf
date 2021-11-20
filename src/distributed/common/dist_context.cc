/*!
 * Copyright (c) 2020 by Contributors
 * \file src/distributed/context.cc
 * \brief Context of Distributed Settings.
 */
#include "mnm/registry.h"
#include "mnm/dist_context.h"

namespace mnm {
namespace distributed {

using communicator::Communicator;
using communicator::CommunicatorManager;

DistContext DistContext::make() {
  ir::ObjectPtr<DistContextObj> n = ir::make_object<DistContextObj>();
  Communicator* comm = CommunicatorManager::Get()->GetCommunicator();
  n->root_rank = comm->GetRootRank();
  n->rank = comm->GetRank();
  n->size = comm->GetSize();
  n->local_rank = comm->GetLocalRank();
  n->local_size = comm->GetLocalSize();

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
