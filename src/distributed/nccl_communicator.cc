/*!
 * Copyright (c) 2019 by Contributors
 * \file src/distributed/nccl_communicator.cc
 * \brief Communicator of NCCL
 */

#include <nccl.h>
#include "mnm/communicator.h"

#define NCCL_CALL(cmd)                                                                            \
  do {                                                                                            \
    ncclResult_t e = cmd;                                                                         \
    if (e != ncclSuccess) {                                                                       \
      LOG(INFO) << "Failed: NCCL error " << __FILE__ << ":" << __LINE__ << ncclGetErrorString(e); \
      exit(EXIT_FAILURE);                                                                         \
    }                                                                                             \
  } while (0)

namespace mnm {
namespace distributed {
namespace communicator {

class NCCLCommunicator : public Communicator {
 public:
  NCCLCommunicator() {
    Init();
  }
  virtual ~NCCLCommunicator() {
    Finalize();
  }
  virtual void Init() {
    GetConnector();
    cudaSetDevice(GetLocalRank());
    if (GetRank() == root_rank) {
      NCCL_CALL(ncclGetUniqueId(&nccl_id));
    }
    connector_->Broadcast(reinterpret_cast<void*>(&nccl_id), sizeof(nccl_id), root_rank);
    NCCL_CALL(ncclCommInitRank(&nccl_comm, GetSize(), nccl_id, GetRank()));
  }
  virtual void Finalize() {
    NCCL_CALL(ncclCommDestroy(nccl_comm));
  }
  virtual void* GetCommHandle() {
    return nccl_comm;
  }
  static void* make() {
    return new NCCLCommunicator();
  }

 public:
  std::string type = "NCCL";

 private:
  ncclUniqueId nccl_id;
  ncclComm_t nccl_comm;
};

MNM_REGISTER_GLOBAL("mnm.distributed.communicator._make.nccl")
    .set_body_typed(NCCLCommunicator::make);

int GetRootRank() {
  Communicator* comm = CommunicatorManager::Get()->GetCommunicator();
  return comm->root_rank;
}

int GetRank() {
  Communicator* comm = CommunicatorManager::Get()->GetCommunicator();
  return comm->GetRank();
}

int GetLocalRank() {
  Communicator* comm = CommunicatorManager::Get()->GetCommunicator();
  return comm->GetLocalRank();
}

int GetSize() {
  Communicator* comm = CommunicatorManager::Get()->GetCommunicator();
  return comm->GetSize();
}

int GetLocalSize() {
  Communicator* comm = CommunicatorManager::Get()->GetCommunicator();
  return comm->GetLocalSize();
}

void Synchronize() {
  cudaDeviceSynchronize();
}

void RemoveCommunicator() {
  CommunicatorManager::Get()->Remove();
}

MNM_REGISTER_GLOBAL("mnm.distributed.GetRootRank").set_body_typed(GetRootRank);
MNM_REGISTER_GLOBAL("mnm.distributed.GetRank").set_body_typed(GetRank);
MNM_REGISTER_GLOBAL("mnm.distributed.GetLocalRank").set_body_typed(GetLocalRank);
MNM_REGISTER_GLOBAL("mnm.distributed.GetSize").set_body_typed(GetSize);
MNM_REGISTER_GLOBAL("mnm.distributed.GetLocalSize").set_body_typed(GetLocalSize);
MNM_REGISTER_GLOBAL("mnm.distributed.Synchronize").set_body_typed(Synchronize);
MNM_REGISTER_GLOBAL("mnm.distributed.RemoveCommunicator").set_body_typed(RemoveCommunicator);

}  // namespace communicator
}  // namespace distributed
}  // namespace mnm
