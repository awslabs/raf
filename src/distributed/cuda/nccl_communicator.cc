/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/distributed/nccl_communicator.cc
 * \brief Communicator of NCCL
 */

#include <nccl.h>
#include "raf/communicator.h"

#define NCCL_CALL(cmd)                                                                            \
  do {                                                                                            \
    ncclResult_t e = cmd;                                                                         \
    if (e != ncclSuccess) {                                                                       \
      LOG(INFO) << "Failed: NCCL error " << __FILE__ << ":" << __LINE__ << ncclGetErrorString(e); \
      exit(EXIT_FAILURE);                                                                         \
    }                                                                                             \
  } while (0)

namespace raf {
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
    if (IsRoot()) {
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

RAF_REGISTER_GLOBAL("raf.distributed.communicator._make.nccl")
    .set_body_typed(NCCLCommunicator::make);

void Synchronize() {
  cudaDeviceSynchronize();
}

RAF_REGISTER_GLOBAL("raf.distributed.Synchronize").set_body_typed(Synchronize);

}  // namespace communicator
}  // namespace distributed
}  // namespace raf
