/*!
 * Copyright (c) 2022 by Contributors
 * \file src/distributed/mpi_communicator.cc
 * \brief MPI Communicator.
 */
#pragma once
#include <mpi.h>
#include "mnm/communicator.h"
#include <string>
#include <functional>

namespace mnm {
namespace distributed {
namespace communicator {

class MPICommunicatorObj final : public CommunicatorObj {
 public:
  const MPI_Comm mpi_comm = MPI_COMM_WORLD;
  static constexpr const char* _type_key = "mnm.distributed.MPICommunicator";
  MNM_FINAL_OBJECT(MPICommunicatorObj, CommunicatorObj);
};

class MPICommunicator final : public Communicator {
 public:
  static MPICommunicator make(TupleValue rank_list);
  virtual ~MPICommunicator();
  MNM_OBJECT_REF(MPICommunicator, Communicator, MPICommunicatorObj);
};

}  // namespace connector
}  // namespace distributed
}  // namespace mnm
