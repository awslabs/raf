/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file src/op/declare/stream.cc
 * \brief Declaration of CUDA stream controlling operators, e.g. synchronize.
 */
#include "raf/op.h"
#include "raf/value.h"
#include "raf/tensor.h"
#include "../schema/stream.h"
#include "./declare_utils.h"

namespace raf {
namespace op {
namespace declare {

using namespace raf::op::schema;
using namespace raf::value;

RAF_OP_DECLARE("raf.op.set_stream",
               [](const CallValues& call) {
                 const auto* args = call->args.as<SetStreamArgs>();
                 CHECK(args != nullptr);
                 call->callee = ir::NullValue<OpValue>();
               })
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TRAFSideEffect>("TRAFSideEffect", true);

RAF_OP_DECLARE("raf.op.add_event",
               [](const CallValues& call) {
                 const auto* args = call->args.as<EventArgs>();
                 CHECK(args != nullptr);
                 call->callee = ir::NullValue<OpValue>();
               })
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TRAFSideEffect>("TRAFSideEffect", true);

RAF_OP_DECLARE("raf.op.wait_event",
               [](const CallValues& call) {
                 const auto* args = call->args.as<EventArgs>();
                 CHECK(args != nullptr);
                 call->callee = ir::NullValue<OpValue>();
               })
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TRAFSideEffect>("TRAFSideEffect", true);

RAF_OP_DECLARE("raf.op.stream_barrier",
               [](const CallValues& call) {
                 const auto* args = call->args.as<StreamBarrierArgs>();
                 CHECK(args != nullptr);
                 call->callee = ir::NullValue<OpValue>();
               })
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TRAFSideEffect>("TRAFSideEffect", true);

RAF_OP_DECLARE("raf.op.stream_sync", [](const CallValues& call) {
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
}  // namespace raf
