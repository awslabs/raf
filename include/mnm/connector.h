/*!
 * Copyright (c) 2019 by Contributors
 * \file connector.h
 * \brief Connector resources.
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

namespace mnm {
namespace distributed {
namespace connector {

using registry::GetPackedFunc;

class Connector {
 public:
  Connector() {
  }
  virtual ~Connector() {
  }
  virtual void Init() = 0;
  virtual void Broadcast(void* buffer, int count, int root) = 0;
  virtual void Barrier() = 0;
  virtual void Finalize() = 0;

 public:
  std::string type;
  int local_size = 0;
  int local_rank = 0;
  int size;
  int rank;
};

class ConnectorManager {
 public:
  // TODO: support multiple connectors.
  ConnectorManager() {
    conn_ = nullptr;
  }
  static ConnectorManager* Get() {
    static ConnectorManager* instance = new ConnectorManager();
    return instance;
  }

  Connector* GetConnector(const std::string& name) {
    CHECK_LT(name.size(), 128) << "There is no such connector: " << name;
    thread_local char maker_name[128];
    if (conn_ == nullptr) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (conn_ == nullptr) {
        // ok, it is truly a nullptr
        if (name == "") {
          const std::string& default_name = "mpi";
          snprintf(maker_name, sizeof(maker_name), "mnm.distributed.connector._make.%s",
                   default_name.c_str());
        } else {
          CHECK_EQ(name, "mpi") << "Unsupported connector: " << name;
          snprintf(maker_name, sizeof(maker_name), "mnm.distributed.connector._make.%s",
                   name.c_str());
        }
        void* ret = GetPackedFunc(maker_name)();
        conn_.reset(static_cast<Connector*>(ret));
        return conn_.get();
      }
    }
    // otherwise this is not nullptr
    CHECK_EQ(name, "") << "You have already initialized a connector [" << conn_->type
                       << "], and currently we do not support multiple connectors";
    return conn_.get();
  }

  void Remove() {
    std::lock_guard<std::mutex> lock(mutex_);
    conn_ = nullptr;
  }

 public:
  std::shared_ptr<Connector> conn_;
  std::mutex mutex_;
};

}  // namespace connector
}  // namespace distributed
}  // namespace mnm
