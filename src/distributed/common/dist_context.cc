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

MNM_REGISTER_GLOBAL("mnm.distributed._make.DistContext").set_body_typed(DistContext::make);
MNM_REGISTER_GLOBAL("mnm.distributed.Global").set_body_typed(DistContext::Global);

MNM_REGISTER_OBJECT_REFLECT(DistContextObj);

}  // namespace distributed
}  // namespace mnm
