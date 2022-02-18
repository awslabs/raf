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

MNM_REGISTER_GLOBAL("mnm.distributed.communicator._make.nccl")
    .set_body_typed(NCCLCommunicator::make);

void Synchronize() {
  cudaDeviceSynchronize();
}

MNM_REGISTER_GLOBAL("mnm.distributed.Synchronize").set_body_typed(Synchronize);

}  // namespace communicator
}  // namespace distributed
}  // namespace mnm
