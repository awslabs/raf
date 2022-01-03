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
  virtual void Broadcast(void* buffer, int count, int root) = 0;
  virtual void Barrier() = 0;

 public:
  std::string type;
  std::vector<uint64_t> host_ids;
  int local_size = 0;
  int local_rank = 0;
  int size = 1;
  int rank = 0;
};

class ConnectorManager {
 public:
  ConnectorManager() {
  }
  static ConnectorManager* Get() {
    static ConnectorManager* instance = new ConnectorManager();
    return instance;
  }

  Connector* GetConnector(const std::string& name = "mpi") {
    if (conn_.count(name) == 0) {
      const std::string prefix = "mnm.distributed.connector._make.";
      void* conn_handler =
          GetPackedFunc(prefix + name)();   // will check whether the function exists or not
      std::shared_ptr<Connector> conn_ptr;  // NOTE: should we return a shared_ptr<Conn>?
      conn_ptr.reset(static_cast<Connector*>(conn_handler));
      conn_[name] = std::move(conn_ptr);
    }
    return conn_[name].get();
  }

  void Remove() {
    conn_.clear();
  }

 public:
  std::map<std::string, std::shared_ptr<Connector>> conn_;
};

}  // namespace connector
}  // namespace distributed
}  // namespace mnm
