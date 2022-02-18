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
 * \file src/op/declare/stream.cc
 * \brief Declaration of CUDA stream controlling operators, e.g. synchronize.
 */
#include "mnm/op.h"
#include "mnm/value.h"
#include "mnm/tensor.h"
#include "../schema/stream.h"
#include "./declare_utils.h"

namespace mnm {
namespace op {
namespace declare {

using namespace mnm::op::schema;
using namespace mnm::value;

MNM_OP_DECLARE("mnm.op.set_stream",
               [](const CallValues& call) {
                 const auto* args = call->args.as<SetStreamArgs>();
                 CHECK(args != nullptr);
                 call->callee = ir::NullValue<OpValue>();
               })
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TMNMSideEffect>("TMNMSideEffect", true);

MNM_OP_DECLARE("mnm.op.add_event",
               [](const CallValues& call) {
                 const auto* args = call->args.as<EventArgs>();
                 CHECK(args != nullptr);
                 call->callee = ir::NullValue<OpValue>();
               })
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TMNMSideEffect>("TMNMSideEffect", true);

MNM_OP_DECLARE("mnm.op.wait_event",
               [](const CallValues& call) {
                 const auto* args = call->args.as<EventArgs>();
                 CHECK(args != nullptr);
                 call->callee = ir::NullValue<OpValue>();
               })
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TMNMSideEffect>("TMNMSideEffect", true);

MNM_OP_DECLARE("mnm.op.stream_barrier",
               [](const CallValues& call) {
                 const auto* args = call->args.as<StreamBarrierArgs>();
                 CHECK(args != nullptr);
                 call->callee = ir::NullValue<OpValue>();
               })
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TMNMSideEffect>("TMNMSideEffect", true);

MNM_OP_DECLARE("mnm.op.stream_sync", [](const CallValues& call) {
  const auto* args = call->args.as<StreamArgs>();
  CHECK(args != nullptr);
  auto& tag = args->stream_tag;
  auto& data = args->x;
  const DLTensor* x = data;
  call->device = x->device;
  call->out = data;
});

}  // namespace declare
}  // namespace op
}  // namespace mnm
