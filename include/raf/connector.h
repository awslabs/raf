/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
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
#include "raf/registry.h"
#include "raf/value.h"

namespace raf {
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
  int size = 1;
  int rank = 0;
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

    std::string default_name = "mpi";
    snprintf(maker_name, sizeof(maker_name), "raf.distributed.connector._make.%s",
             default_name.c_str());
    const registry::PackedFunc* pf = registry::Registry::Get(maker_name);
    if (pf == nullptr) default_name = "void";

    if (conn_ == nullptr) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (conn_ == nullptr) {
        // ok, it is truly a nullptr
        if (name == "") {
          snprintf(maker_name, sizeof(maker_name), "raf.distributed.connector._make.%s",
                   default_name.c_str());
        } else {
          if (name != "void") CHECK_EQ(name, "mpi") << "Unsupported connector: " << name;
          snprintf(maker_name, sizeof(maker_name), "raf.distributed.connector._make.%s",
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
}  // namespace raf
