/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/distributed/nccl_communicator.cc
 * \brief NCCL Communicator.
 */

#include "raf/nccl_communicator.h"

namespace raf {
namespace distributed {
namespace communicator {

NCCLCommunicatorObj::~NCCLCommunicatorObj() {
  NCCL_CALL(ncclCommDestroy(nccl_comm));
}

NCCLCommunicator NCCLCommunicator::make(value::TupleValue rank_list) {
  auto mpi = Communicator::Get("mpi");  // Must init MPI first
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
    MPI_CALL(MPI_Bcast(reinterpret_cast<void*>(&nccl_id), sizeof(nccl_id), MPI_BYTE, obj->root_rank,
                       MPI_COMM_WORLD));
    NCCL_CALL(ncclCommInitRank(&obj->nccl_comm, obj->size, nccl_id, obj->rank));
  } else {
    // Create Sub-communicator
    // ALL the nodes including nodes not in the rank_list MUST join the process of creating this
    // sub-communicator due to MPI_Bcast. If this rank is not in rank_list, this communicator will
    // run in standalone mode.
    InitSubCommunicator(NCCLCommunicator(obj), rank_list, mpi);
    obj->parent_comm = mpi;
    MPI_CALL(MPI_Bcast(reinterpret_cast<void*>(&nccl_id), sizeof(nccl_id), MPI_BYTE, obj->root_rank,
                       MPI_COMM_WORLD));
    NCCL_CALL(ncclCommInitRank(&obj->nccl_comm, obj->size, nccl_id, obj->rank));
  }

  return NCCLCommunicator(obj);
}

RAF_REGISTER_GLOBAL("raf.distributed.communicator._make.nccl")
    .set_body_typed(NCCLCommunicator::make);

}  // namespace communicator
}  // namespace distributed
}  // namespace raf
