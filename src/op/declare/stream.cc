/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
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
