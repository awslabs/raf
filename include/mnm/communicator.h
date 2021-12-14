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
  Communicator() {
  }
  virtual ~Communicator() {
  }
  int GetLocalSize() {
    return connector_->local_size;
  }
  int GetLocalRank() {
    return connector_->local_rank;
  }
  int GetSize() {
    return connector_->size;
  }
  int GetRank() {
    return connector_->rank;
  }
  int GetRootRank() {
    return root_rank;
  }
  bool IsRoot() {
    return GetRank() == GetRootRank();
  }
  virtual void* GetCommHandle() = 0;

 protected:
  virtual void Init() = 0;
  virtual void Finalize() = 0;
  void GetConnector(const std::string& name = "mpi") {
    connector_.reset(ConnectorManager::Get()->GetConnector(name));
  }

 public:
  std::string type;
  int root_rank = 0;
  std::shared_ptr<Connector> connector_;
};

class CommunicatorManager {
 public:
  CommunicatorManager() {
  }

  static CommunicatorManager* Get() {
    static CommunicatorManager* instance = new CommunicatorManager();
    return instance;
  }

  Communicator* GetCommunicator(const std::string& name = "nccl", const std::vector<int64_t>& rank_list = {}) {
    std::lock_guard<std::mutex> lock(mutex_); // writing to std::map is not thread-safe
    auto id = CommunicatorID(name, rank_list);
    if (comm_.count(id) == 0) {
      const std::string prefix = "mnm.distributed.communicator._make.";
      void* comm_handler = GetPackedFunc(prefix + name)(); // will check whether the function exists or not
      std::shared_ptr<Communicator> comm_ptr; // NOTE: should comm_handler be a unique_ptr to prevent from being used simuatenously?
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
  std::mutex mutex_;
};

}  // namespace communicator
}  // namespace distributed
}  // namespace mnm
