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
    GetConnector();
    cudaSetDevice(GetLocalRank());
    if (IsRoot()) {
      NCCL_CALL(ncclGetUniqueId(&nccl_id));
    }
    connector_->Broadcast(reinterpret_cast<void*>(&nccl_id), sizeof(nccl_id), root_rank);
    NCCL_CALL(ncclCommInitRank(&nccl_comm, GetSize(), nccl_id, GetRank()));
  }
  virtual ~NCCLCommunicator() {
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

void Synchronize() {
  cudaDeviceSynchronize();
}

MNM_REGISTER_GLOBAL("mnm.distributed.Synchronize").set_body_typed(Synchronize);

}  // namespace communicator
}  // namespace distributed
}  // namespace mnm
