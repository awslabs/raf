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

MNM_REGISTER_GLOBAL("mnm.distributed.connector._make.void").set_body_typed(VoidConnector::make);

}  // namespace connector
}  // namespace distributed
}  // namespace mnm
