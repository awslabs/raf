/*!
 * Copyright (c) 2019 by Contributors
 * \file src/distributed/nccl_communicator.cc
 * \brief Communicator of NCCL
 */

#include <algorithm>
#include <nccl.h>
#include "mnm/communicator.h"
#include "mnm/op_utils.h"
#include "mnm/value.h"

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
  NCCLCommunicator(const std::vector<int64_t>& rank_list = {}) {
    GetConnector();
    cudaSetDevice(GetLocalRank());
    ncclUniqueId nccl_id;
    NCCL_CALL(ncclGetUniqueId(&nccl_id));
    if (rank_list.empty()) {
      connector_->Broadcast(reinterpret_cast<void*>(&nccl_id), sizeof(nccl_id), root_rank);
      NCCL_CALL(ncclCommInitRank(&nccl_comm, GetSize(), nccl_id, GetRank()));
    } else {
      auto world_size = GetSize();
      auto world_rank = GetRank();
      int size = rank_list.size();
      int rank;
      CHECK_LE(size, world_size);
      for (rank = 0; rank < size; ++rank) {
        if (rank_list[rank] == world_rank) break;
      }
      connector_->Broadcast(reinterpret_cast<void*>(&nccl_id), sizeof(nccl_id), rank_list[0]);
      if (rank < size) {
        NCCL_CALL(ncclCommInitRank(&nccl_comm, size, nccl_id, rank));
      } else {
        NCCL_CALL(ncclGetUniqueId(&nccl_id));
        NCCL_CALL(ncclCommInitRank(&nccl_comm, 1, nccl_id, 0));
        // ALL the nodes including the irrelevant ones MUST join the process of creating this
        // sub-communicator the irrelevant nodes should not use this communicator though
      }
    }
  }
  virtual ~NCCLCommunicator() {
    NCCL_CALL(ncclCommDestroy(nccl_comm));
  }
  virtual void* GetCommHandle() {
    return nccl_comm;
  }
  static void* make(value::TupleValue obj) {
    std::vector<int64_t> rank_list;
    for (auto i : obj->fields) {
      auto val = Downcast<value::IntValue>(i);
      rank_list.push_back(val->value);
    }
    return new NCCLCommunicator(rank_list);
  }

 public:
  std::string type = "NCCL";

 private:
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
