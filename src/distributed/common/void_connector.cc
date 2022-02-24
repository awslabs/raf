/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/distributed/void_connector.cc
 * \brief A void Connector used as a template
 */

#include "raf/connector.h"

namespace raf {
namespace distributed {
namespace connector {

class VoidConnector : public Connector {
 public:
  VoidConnector() {
    Init();
  }
  virtual ~VoidConnector() {
    Finalize();
  }
  virtual void Init() {
    LOG(INFO) << "You have created a VoidConnector, which will do nothing and can not be used for "
                 "parallel training.";
  }
  virtual void Broadcast(void* buffer, int count, int root) {
  }
  virtual void Barrier() {
  }
  virtual void Finalize() {
  }

 public:
  static void* make() {
    return new VoidConnector();
  }

 public:
  std::string type = "VOID";
};

RAF_REGISTER_GLOBAL("raf.distributed.connector._make.void").set_body_typed(VoidConnector::make);

}  // namespace connector
}  // namespace distributed
}  // namespace raf
