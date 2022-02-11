/*!
 * Copyright (c) 2022 by Contributors
 * \file src/distributed/nccl_communicator.cc
 * \brief NCCL Communicator
 */

#include "mnm/nccl_communicator.h"

namespace mnm {
namespace distributed {
namespace communicator {

NCCLCommunicator NCCLCommunicator::make(value::TupleValue rank_list)  {
  auto mpi = Communicator::Get("mpi"); // Must init MPI first
  auto obj = make_object<NCCLCommunicatorObj>();

  ncclUniqueId nccl_id;
  NCCL_CALL(ncclGetUniqueId(&nccl_id));

  if (rank_list->fields.empty()) {
    // Create Global Communicator
    obj->local_size = mpi->local_size;
    obj->local_rank = mpi->local_rank;
    obj->size = mpi->size;
    obj->rank = mpi->rank;
    obj->world_size = mpi->world_size;
    obj->world_rank = mpi->world_rank;
    obj->root_rank = mpi->root_rank;
    obj->host_ids = mpi->host_ids;
    obj->parent_comm = mpi;
    cudaSetDevice(obj->local_rank);
    MPI_CALL(MPI_Bcast(reinterpret_cast<void*>(&nccl_id), sizeof(nccl_id), MPI_BYTE, obj->root_rank, MPI_COMM_WORLD));
    NCCL_CALL(ncclCommInitRank(&obj->nccl_comm, obj->size, nccl_id, obj->rank));
  } else {
    // Create Sub-communicator
    // ALL the nodes including the irrelevant ones MUST join the process of creating this
    // sub-communicator. When this rank is not in rank_list, it will run in standalone mode.
    InitSubCommunicator(NCCLCommunicator(obj), rank_list, mpi);
    obj->parent_comm = mpi;
    MPI_CALL(MPI_Bcast(reinterpret_cast<void*>(&nccl_id), sizeof(nccl_id), MPI_BYTE, obj->root_rank, MPI_COMM_WORLD));
    NCCL_CALL(ncclCommInitRank(&obj->nccl_comm, obj->size, nccl_id, obj->rank));
  }

  return NCCLCommunicator(obj);
}

NCCLCommunicator::~NCCLCommunicator() {
  NCCL_CALL(ncclCommDestroy(operator->()->nccl_comm));
}

MNM_REGISTER_GLOBAL("mnm.distributed.communicator._make.nccl")
    .set_body_typed(NCCLCommunicator::make);

}  // namespace communicator
}  // namespace distributed
}  // namespace mnm
