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
 * \file src/distributed/mpi_communicator.cc
 * \brief MPI Communicator.
 */

#include "mnm/mpi_communicator.h"

namespace mnm {
namespace distributed {
namespace communicator {

MPICommunicatorObj::~MPICommunicatorObj() {
  MPI_CALL(MPI_Finalize());
}

MPICommunicator MPICommunicator::make(TupleValue rank_list) {
  CHECK(rank_list->fields.empty())
      << "MPICommunicator doesn't support creating a sub-communicator yet.";
  auto obj = make_object<MPICommunicatorObj>();

  int initialized = 0;
  MPI_CALL(MPI_Initialized(&initialized));
  if (initialized) return NullValue<MPICommunicator>();

  int rank, size;
  MPI_CALL(MPI_Init(nullptr, nullptr));
  MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &size));

  obj->host_ids.resize(size);

  obj->host_ids[rank] = GetHostID();
  // Allgather the hostIDs of nodes.
  MPI_CALL(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &obj->host_ids[0], sizeof(uint64_t),
                         MPI_BYTE, MPI_COMM_WORLD));

  int local_rank = 0, local_size = 0;
  // Get local rank
  for (int p = 0; p < size; ++p) {
    if (p == rank) break;
    if (obj->host_ids[p] == obj->host_ids[rank]) local_rank++;
  }
  // Get local size
  for (int p = 0; p < size; ++p) {
    if (obj->host_ids[p] == obj->host_ids[rank]) local_size++;
  }
  obj->local_size = local_size;
  obj->local_rank = local_rank;
  obj->size = size;
  obj->rank = rank;
  obj->world_size = size;
  obj->world_rank = rank;
  obj->root_rank = 0;
  return MPICommunicator(obj);
}

MNM_REGISTER_GLOBAL("mnm.distributed.communicator._make.mpi").set_body_typed(MPICommunicator::make);

}  // namespace communicator
}  // namespace distributed
}  // namespace mnm
