/*!
 * Copyright (c) 2020 by Contributors
 * \file src/distributed/void_connector.cc
 * \brief A void Connector used as a template
 */

#include "mnm/connector.h"

namespace mnm {
namespace distributed {
namespace connector {

class VoidConnector : public Connector {
 public:
  VoidConnector() {
    LOG(INFO) << "You have created a VoidConnector, which will do nothing and can not be used for "
                 "parallel training.";
  }
  virtual void Broadcast(void* buffer, int count, int root) {
  }
  virtual void Barrier() {
  }

 public:
  static void* make() {
    return new VoidConnector();
  }

 public:
  std::string type = "VOID";
};

MNM_REGISTER_GLOBAL("mnm.distributed.connector._make.void").set_body_typed(VoidConnector::make);

}  // namespace connector
}  // namespace distributed
}  // namespace mnm
