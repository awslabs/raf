/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/distributed/mpi_connector.cc
 * \brief Connector of MPI.
 */

#include <mpi.h>
#include "mnm/connector.h"

#define MPI_CALL(cmd)                                                         \
  do {                                                                        \
    int e = cmd;                                                              \
    if (e != MPI_SUCCESS) {                                                   \
      LOG(FATAL) << "Failed: MPI error " << __FILE__ << ":" << __LINE__ << e; \
    }                                                                         \
  } while (0)

namespace mnm {
namespace distributed {
namespace connector {

static void GetHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i = 0; i < maxlen; i++) {
    if (hostname[i] == '.') {
      hostname[i] = '\0';
      return;
    }
  }
}

static uint64_t GetHostHash(const char* string) {
  uint64_t result = 5381;
  for (int i = 0; string[i] != '\0'; i++) {
    result = ((result << 5) + result) + string[i];
  }
  return result;
}

class MPIConnector : public Connector {
 public:
  MPIConnector() {
    Init();
  }
  virtual ~MPIConnector() {
    Finalize();
  }
  virtual void Init() {
    int initialized = 0;
    MPI_CALL(MPI_Initialized(&initialized));
    if (initialized) {
      return;
    }

    MPI_CALL(MPI_Init(nullptr, nullptr));

    MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &size));

    std::vector<uint64_t> hostHashs(size);
    char hostname[1024];
    GetHostName(hostname, 1024);
    hostHashs[rank] = GetHostHash(hostname);
    // Allgather the hostHashs of nodes.
    MPI_CALL(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &hostHashs[0], sizeof(uint64_t),
                           MPI_BYTE, MPI_COMM_WORLD));

    // Get local rank
    for (int p = 0; p < size; ++p) {
      if (p == rank) break;
      if (hostHashs[p] == hostHashs[rank]) local_rank++;
    }
    // Get local size
    for (int p = 0; p < size; ++p) {
      if (hostHashs[p] == hostHashs[rank]) local_size++;
    }
  }
  virtual void Broadcast(void* buffer, int count, int root) {
    MPI_CALL(MPI_Bcast(buffer, count, MPI_BYTE, root, MPI_COMM_WORLD));
  }
  virtual void Barrier() {
    MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
  }
  virtual void Finalize() {
    MPI_CALL(MPI_Finalize());
  }

 public:
  static void* make() {
    return new MPIConnector();
  }

 public:
  std::string type = "MPI";
};

MNM_REGISTER_GLOBAL("mnm.distributed.connector._make.mpi").set_body_typed(MPIConnector::make);

}  // namespace connector
}  // namespace distributed
}  // namespace mnm
