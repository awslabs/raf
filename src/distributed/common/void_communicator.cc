/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/distributed/void_communicator.cc
 * \brief A void Communicator used as a template
 */

#include "mnm/communicator.h"

namespace mnm {
namespace distributed {
namespace communicator {

class VoidCommunicator : public Communicator {
 public:
  VoidCommunicator() {
    Init();
  }
  virtual ~VoidCommunicator() {
    Finalize();
  }
  virtual void Init() {
    // In this method, you should
    // 1. Get a connector by calling GetConnector()
    // 2. Create a new communicator and store its handle.
    // 3. Initialize the new communicator.
    GetConnector("void");

    LOG(INFO) << "You have created a VoidCommunicator, which will do nothing and can not be used "
                 "for parallel training.";
  }
  virtual void Finalize() {
  }
  virtual void* GetCommHandle() {
    return void_comm_handle;
  }
  static void* make() {
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
}

MNM_REGISTER_GLOBAL("mnm.distributed.RemoveCommunicator").set_body_typed(RemoveCommunicator);

}  // namespace communicator
}  // namespace distributed
}  // namespace mnm
