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

void OverlapCommForward(bool overlap) {
  DistContext::Global()->overlap_comm_forward = overlap;
}

void AudoDPProfilingStartIter(int auto_dp_profiling_start_iter) {
  DistContext::Global()->auto_dp_profiling_start_iter = auto_dp_profiling_start_iter;
}

void AudoDPProfilingEndIter(int auto_dp_profiling_end_iter) {
  DistContext::Global()->auto_dp_profiling_end_iter = auto_dp_profiling_end_iter;
}

MNM_REGISTER_GLOBAL("mnm.distributed._make.DistContext").set_body_typed(DistContext::make);
MNM_REGISTER_GLOBAL("mnm.distributed.Global").set_body_typed(DistContext::Global);
MNM_REGISTER_GLOBAL("mnm.distributed.EnableDataParallel").set_body_typed(EnableDataParallel);
MNM_REGISTER_GLOBAL("mnm.distributed.OverlapCommForward").set_body_typed(OverlapCommForward);
MNM_REGISTER_GLOBAL("mnm.distributed.AudoDPProfilingStartIter")
    .set_body_typed(AudoDPProfilingStartIter);
MNM_REGISTER_GLOBAL("mnm.distributed.AudoDPProfilingEndIter")
    .set_body_typed(AudoDPProfilingEndIter);

MNM_REGISTER_OBJECT_REFLECT(DistContextObj);

}  // namespace distributed
}  // namespace mnm
