/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file ./src/op/from_relay/vision.cc
 * \brief Operators bridged from Relay.
 */
#include "raf/op_utils.h"
#include "tvm/relay/attrs/vision.h"
#include "./from_relay_utils.h"

namespace raf {
namespace op {
namespace from_relay {

RAF_OP_FROM_RELAY("vision.roi_align", "raf.op.roi_align",
                  [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {
                    Array<Expr> raf_args = args;
                    const auto* relay_attrs = attrs.as<ROIAlignAttrs>();
                    raf_args.push_back(MakeConstant(ArrayToIntTuple(relay_attrs->pooled_size)));
                    raf_args.push_back(MakeConstant(ScalarValue::make(relay_attrs->spatial_scale)));
                    raf_args.push_back(MakeConstant(ScalarValue::make(relay_attrs->sample_ratio)));
                    raf_args.push_back(MakeConstant(StringValue::make(relay_attrs->layout)));
                    raf_args.push_back(MakeConstant(StringValue::make(relay_attrs->mode)));
                    return raf_args;
                  });

}  // namespace from_relay
}  // namespace op
}  // namespace raf
