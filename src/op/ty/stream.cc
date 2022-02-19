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
 * \file src/op/ty/stream.cc
 * \brief Typing relations of cuda stream operators
 */
#include <tvm/relay/type.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/env_func.h>
#include "mnm/type.h"
#include "../schema/stream.h"
#include "./utils.h"

namespace mnm {
namespace op {

using namespace mnm::ir;
using namespace mnm::value;
using namespace mnm::op::schema;

Type StreamEventInfer(const CallValues& value) {
  return TupleType::Empty();
}

Type StreamInfer(const CallValues& value) {
  const auto* args = value->args.as<StreamArgs>();
  CHECK(args != nullptr);
  return GetType(args->x);
}

MNM_OP_TYPE("mnm.op.set_stream", "StreamSwitch", StreamEventInfer);
MNM_OP_TYPE("mnm.op.add_event", "EventAdd", StreamEventInfer);
MNM_OP_TYPE("mnm.op.wait_event", "EventWait", StreamEventInfer);
MNM_OP_TYPE("mnm.op.stream_barrier", "StreamBarrier", StreamEventInfer);
MNM_OP_TYPE("mnm.op.stream_sync", "StreamSync", StreamInfer);

}  // namespace op
}  // namespace mnm
