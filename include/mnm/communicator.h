/*!
 * Copyright (c) 2019 by Contributors
 * \file communicator.h
 * \brief Communication resources.
 */
#pragma once
#include <unistd.h>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <memory>
#include "dmlc/logging.h"
#include "mnm/registry.h"
#include "mnm/value.h"
#include "mnm/op_utils.h"
#include "./connector.h"

typedef std::pair<std::string, std::vector<int64_t>> CommunicatorID;

namespace mnm {
namespace distributed {
namespace communicator {

using connector::Connector;
using connector::ConnectorManager;
using registry::GetPackedFunc;

struct DistAttrs {
  int local_size;
  int local_rank;
  int size;
  int rank;
  int world_size;
  int world_rank;
};

class Communicator {
 public:
  Communicator(const std::vector<int64_t>& rank_list = {}) {
  }
  virtual ~Communicator() {
  }
  int GetLocalSize() {
    return local_size;
  }
  int GetLocalRank() {
    return local_rank;
  }
  int GetSize() {
    return size;
  }
  int GetRank() {
    return rank;
  }
  int GetRootRank() {
    return root_rank;
  }
  static DistAttrs GetDistAttrs(const std::vector<int64_t>& rank_list = {}) {
    auto mpi = ConnectorManager::Get()->GetConnector("mpi");
    if (rank_list.empty()) {
      DistAttrs ret = {.local_size = mpi->local_size,
                       .local_rank = mpi->local_rank,
                       .size = mpi->size,
                       .rank = mpi->rank,
                       .world_size = mpi->size,
                       .world_rank = mpi->rank};
      return ret;
    } else {
      int size = rank_list.size();
      int rank;
      int local_size = 0;
      int local_rank = 0;
      std::vector<int> host_ids;
      CHECK_LE(size, mpi->size);
      for (rank = 0; rank < size; ++rank) {
        if (rank_list[rank] == mpi->rank) break;
      }
      if (rank == size) {
        // This rank is not in rank_list
        rank = -1;
        size = -1;
      }
      for (auto i : rank_list) {
        host_ids.push_back(mpi->host_ids[i]);
      }
      for (int p = 0; p < size; ++p) {
        if (p == rank) break;
        if (host_ids[p] == host_ids[rank]) local_rank++;
      }
      for (int p = 0; p < size; ++p) {
        if (host_ids[p] == host_ids[rank]) local_size++;
      }
      DistAttrs ret = {.local_size = local_size,
                       .local_rank = local_rank,
                       .size = size,
                       .rank = rank,
                       .world_size = mpi->size,
                       .world_rank = mpi->rank};
      return ret;
    }
  }

  virtual void* GetCommHandle() = 0;

 public:
  std::string type;
  int root_rank = 0;
  int local_size = 0;
  int local_rank = 0;
  int size = 1;
  int rank = 0;
};

class CommunicatorManager {
 public:
  CommunicatorManager() {
  }

  static CommunicatorManager* Get() {
    static CommunicatorManager* instance = new CommunicatorManager();
    return instance;
  }

  Communicator* GetCommunicator(const std::string& name = "",
                                const std::vector<int64_t>& rank_list = {}) {
    auto id = CommunicatorID(name, rank_list);

    if (comm_.count(id) == 0) {
      const std::string prefix = "mnm.distributed.communicator._make.";
      auto func_name = prefix + (!name.empty() ? name :
#ifdef MNM_USE_NCCL
                                               "nccl"
#else
                                               "void"
#endif
                                );
      void* comm_handler = GetPackedFunc(func_name)(
          op::ArrayToIntTuple(rank_list));     // will check whether the function exists or not
      std::shared_ptr<Communicator> comm_ptr;  // NOTE: should we return a shared_ptr<Comm>?
      comm_ptr.reset(static_cast<Communicator*>(comm_handler));
      comm_[id] = std::move(comm_ptr);
    }
    return comm_[id].get();
  }

  void Remove() {
    comm_.clear();
  }

 public:
  std::map<CommunicatorID, std::shared_ptr<Communicator>> comm_;
};

}  // namespace communicator
}  // namespace distributed
}  // namespace mnm
