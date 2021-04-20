/*!
 * Copyright (c) 2019 by Contributors
 * \file src/op/declare/stream_control.cc
 * \brief Declaration of stream controlling operators, e.g. synchronize.
 */
#include "mnm/op.h"
#include "mnm/value.h"
#include "mnm/tensor.h"
#include "../schema/communication.h"
#include "../schema/ufunc.h"
#include "./declare_utils.h"

namespace mnm {
namespace op {
namespace declare {

using namespace mnm::op::schema;
using namespace mnm::value;
using tensor::Tensor;

void StreamSync(const CallValues& call) {
  const auto* args = call->args.as<StreamControlArgs>();
  CHECK(args != nullptr);
  auto& tag = args->stream_tag;
  auto& data = args->x;
  const DLTensor* x = data;
  call->device = x->device;
  call->out = VoidValue::make();
}

MNM_OP_DECLARE("mnm.op.stream_sync", StreamSync);

}  // namespace declare
}  // namespace op
}  // namespace mnm
