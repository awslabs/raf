/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
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
