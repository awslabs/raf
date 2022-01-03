/*!
 * Copyright (c) 2020 by Contributors
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

static uint64_t GetHostID() {
  char data[1024];
  uint64_t posix_hostid =
      gethostid();  // Prevent confusion if all the nodes share the same hostname
  snprintf(data, 17, "%016lx", posix_hostid);
  gethostname(&data[16], 1000);

  // Bernstein hash
  uint64_t result = 5381;
  for (int i = 0; data[i] != '\0'; i++) {
    result = ((result << 5) + result) + data[i];
  }
  return result;
}

class MPIConnector : public Connector {
 public:
  MPIConnector() {
    int initialized = 0;
    MPI_CALL(MPI_Initialized(&initialized));
    if (initialized) return;

    MPI_CALL(MPI_Init(nullptr, nullptr));

    MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &size));

    host_ids.resize(size);

    host_ids[rank] = GetHostID();
    // Allgather the hostIDs of nodes.
    MPI_CALL(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &host_ids[0], sizeof(uint64_t),
                           MPI_BYTE, MPI_COMM_WORLD));

    // Get local rank
    for (int p = 0; p < size; ++p) {
      if (p == rank) break;
      if (host_ids[p] == host_ids[rank]) local_rank++;
    }
    // Get local size
    for (int p = 0; p < size; ++p) {
      if (host_ids[p] == host_ids[rank]) local_size++;
    }
  }
  virtual ~MPIConnector() {
    MPI_CALL(MPI_Finalize());
  }
  virtual void Broadcast(void* buffer, int count, int root) {
    MPI_CALL(MPI_Bcast(buffer, count, MPI_BYTE, root, MPI_COMM_WORLD));
  }
  virtual void Barrier() {
    MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
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
