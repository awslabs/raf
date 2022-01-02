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

  Communicator* GetCommunicator(const std::string& name = "nccl",
                                const std::vector<int64_t>& rank_list = {}) {
    auto id = CommunicatorID(name, rank_list);
    if (comm_.count(id) == 0) {
      const std::string prefix = "mnm.distributed.communicator._make.";
      void* comm_handler = GetPackedFunc(prefix + name)(
          op::ArrayToIntTuple(rank_list));     // will check whether the function exists or not
      std::shared_ptr<Communicator> comm_ptr;  // NOTE: should we return a shared_ptr<Comm>?
      comm_ptr.reset(static_cast<Communicator*>(comm_handler));
      comm_[id] = std::move(comm_ptr);
    }
    return comm_[id].get();
  }

  void Remove() {
    LOG_ERROR << "CommunicatorManager::Remove is not implemented yet.";
  }

 public:
  std::map<CommunicatorID, std::shared_ptr<Communicator>> comm_;
};

}  // namespace communicator
}  // namespace distributed
}  // namespace mnm
