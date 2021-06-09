/*!
 * Copyright (c) 2021 by Contributors
 * \file src/op/declare/stream.cc
 * \brief Declaration of CUDA stream controlling operators, e.g. synchronize.
 */
#include "mnm/op.h"
#include "mnm/value.h"
#include "mnm/tensor.h"
#include "../schema/stream.h"
#include "../schema/ufunc.h"
#include "./declare_utils.h"

namespace mnm {
namespace op {
namespace declare {

using namespace mnm::op::schema;
using namespace mnm::value;
using tensor::Tensor;

void make_stream_op(const CallValues& call) {
  const auto* args = call->args.as<StreamArgs>();
  CHECK(args != nullptr);
  auto& tag = args->stream_tag;
  auto& data = args->x;
  const DLTensor* x = data;
  call->device = x->device;
  call->out = data;
}

void StreamSync(const CallValues& call) {
  make_stream_op(call);
}

MNM_OP_DECLARE("mnm.op.stream_sync", StreamSync);

void StreamStart(const CallValues& call) {
  make_stream_op(call);
}

MNM_OP_DECLARE("mnm.op.stream_start", StreamStart);

void StreamEnd(const CallValues& call) {
  make_stream_op(call);
}

MNM_OP_DECLARE("mnm.op.stream_end", StreamEnd);

void StreamWait(const CallValues& call) {
  make_stream_op(call);
}

MNM_OP_DECLARE("mnm.op.stream_wait", StreamWait);

}  // namespace declare
}  // namespace op
}  // namespace mnm
