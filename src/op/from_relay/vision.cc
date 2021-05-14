/*!
 * Copyright (c) 2021 by Contributors
 * \file ./src/op/from_relay/vision.cc
 * \brief Operators bridged from Relay.
 */
#include "tvm/relay/attrs/vision.h"
#include "./from_relay_utils.h"

namespace mnm {
namespace op {
namespace from_relay {

MNM_OP_FROM_RELAY("vision.roi_align", "mnm.op.roi_align",
                  [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
                    Array<Expr> mnm_args = args;
                    const auto* relay_attrs = attrs.as<ROIAlignAttrs>();
                    mnm_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->pooled_size)));
                    mnm_args.push_back(MakeConstant(ScalarValue::make(relay_attrs->spatial_scale)));
                    mnm_args.push_back(MakeConstant(ScalarValue::make(relay_attrs->sample_ratio)));
                    mnm_args.push_back(MakeConstant(StringValue::make(relay_attrs->layout)));
                    mnm_args.push_back(MakeConstant(StringValue::make(relay_attrs->mode)));
                    return mnm_args;
                  });

}  // namespace from_relay
}  // namespace op
}  // namespace mnm
