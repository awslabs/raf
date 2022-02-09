/*!
 * Copyright (c) 2019 by Contributors
 * \file src/distributed/void_communicator.cc
 * \brief A void Communicator used as a template
 */

#include "mnm/communicator.h"
#include "mnm/connector.h"

namespace mnm {
namespace distributed {
namespace communicator {

class VoidCommunicator : public Communicator {
 public:
  VoidCommunicator(const std::vector<int64_t>& rank_list = {}) {
    // In this method, you should
    // 1. Get a connector by calling GetConnector()
    // 2. Create a new communicator and store its handle.
    // 3. Initialize the new communicator.

    LOG(INFO) << "You have created a VoidCommunicator, which will do nothing and can not be used "
                 "for parallel training.";
  }
  virtual ~VoidCommunicator() {
  }
  virtual void* GetCommHandle() {
    return void_comm_handle;
  }
  static void* make(value::TupleValue obj) {
    return new VoidCommunicator();
  }

 public:
  std::string type = "VOID";

 private:
  void* void_id;           // an identifier of communicator, e.g. nccl_id
  void* void_comm_handle;  // a handle of communicator, e.g. nccl_comm
};

MNM_REGISTER_GLOBAL("mnm.distributed.communicator._make.void")
    .set_body_typed(VoidCommunicator::make);

void RemoveCommunicator() {
  CommunicatorManager::Get()->Remove();
  ConnectorManager::Get()->Remove();
}

MNM_REGISTER_GLOBAL("mnm.distributed.RemoveCommunicator").set_body_typed(RemoveCommunicator);

}  // namespace communicator
}  // namespace distributed
}  // namespace mnm
