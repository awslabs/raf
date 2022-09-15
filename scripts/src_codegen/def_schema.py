# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict

from .codegen_utils import Arg

OptionalIntArray = "ir::Optional<ir::Array<value::IntValue>>"
OptionalValue = "ir::Optional<value::Value>"
OptionalTensor = "ir::Optional<value::BaseTensorValue>"
SCHEMAS = {
    "nn.h::conv": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="w", cxx_type="value::BaseTensorValue"),
        Arg(
            name="stride",
            cxx_type="std::vector<int64_t>",
            cxx_default="{1}",
            py_default=1,
            cxx_normalizer="IntTuple",
        ),
        Arg(
            name="padding",
            cxx_type="std::vector<int64_t>",
            cxx_default="{0}",
            py_default=0,
            cxx_normalizer="IntTuple",
        ),
        Arg(
            name="dilation",
            cxx_type="std::vector<int64_t>",
            cxx_default="{1}",
            py_default=1,
            cxx_normalizer="IntTuple",
        ),
        Arg(name="groups", cxx_type="int64_t", cxx_default=1),
        Arg(name="layout", cxx_type="std::string", cxx_default='"NCHW"', py_default='"NCHW"'),
        Arg(
            name="kernel_layout", cxx_type="std::string", cxx_default='"OIHW"', py_default='"OIHW"'
        ),
        Arg(name="out_layout", cxx_type="std::string", cxx_default='"NCHW"', py_default='"NCHW"'),
    ],
    "nn.h::conv_trans": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="w", cxx_type="value::BaseTensorValue"),
        Arg(
            name="stride",
            cxx_type="std::vector<int64_t>",
            cxx_default="{1}",
            py_default=1,
            cxx_normalizer="IntTuple",
        ),
        Arg(
            name="padding",
            cxx_type="std::vector<int64_t>",
            cxx_default="{0}",
            py_default=0,
            cxx_normalizer="IntTuple",
        ),
        Arg(
            name="output_padding",
            cxx_type="std::vector<int64_t>",
            cxx_default="{0}",
            py_default=0,
            cxx_normalizer="IntTuple",
        ),
        Arg(
            name="dilation",
            cxx_type="std::vector<int64_t>",
            cxx_default="{1}",
            py_default=1,
            cxx_normalizer="IntTuple",
        ),
        Arg(name="groups", cxx_type="int64_t", cxx_default=1),
        Arg(name="layout", cxx_type="std::string", cxx_default='"NCHW"', py_default='"NCHW"'),
        Arg(
            name="kernel_layout", cxx_type="std::string", cxx_default='"IOHW"', py_default='"IOHW"'
        ),
        Arg(name="out_layout", cxx_type="std::string", cxx_default='"NCHW"', py_default='"NCHW"'),
    ],
    "nn.h::pool": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="kernel", cxx_type="std::vector<int64_t>", cxx_normalizer="IntTuple"),
        Arg(name="stride", cxx_type="std::vector<int64_t>", cxx_normalizer="IntTuple"),
        Arg(
            name="padding",
            cxx_type="std::vector<int64_t>",
            cxx_default="{0}",
            py_default=0,
            cxx_normalizer="IntTuple",
        ),
        Arg(
            name="dilation",
            cxx_type="std::vector<int64_t>",
            cxx_default="{1}",
            py_default=1,
            cxx_normalizer="IntTuple",
        ),
        Arg(name="ceil_mode", cxx_type="bool", cxx_default=False),
        Arg(name="include_pad", cxx_type="bool", cxx_default=True),
        Arg(name="layout", cxx_type="std::string", cxx_default='"NCHW"', py_default='"NCHW"'),
    ],
    "nn.h::adaptive_pool": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="shape", cxx_type="std::vector<int64_t>", cxx_normalizer="IntTuple"),
        Arg(name="layout", cxx_type="std::string", cxx_default='"NCHW"', py_default='"NCHW"'),
    ],
    "nn.h::pad": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        # [(w1, w2), ..., ]
        Arg(name="pad_width", cxx_type="std::vector<int64_t>", cxx_normalizer="IntTuple"),
        Arg(name="pad_value", cxx_type="double", cxx_default=0.0),
        Arg(
            name="pad_mode",
            cxx_type="std::string",
            cxx_default='"constant"',
            py_default='"constant"',
        ),
    ],
    "nn.h::softmax": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="axis", cxx_type="int", cxx_default=-1),
    ],
    "nn.h::batch_norm": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="running_mean", cxx_type="value::BaseTensorValue"),
        Arg(name="running_var", cxx_type="value::BaseTensorValue"),
        Arg(name="w", cxx_type="value::BaseTensorValue", cxx_default="nullptr"),
        Arg(name="b", cxx_type="value::BaseTensorValue", cxx_default="nullptr"),
        Arg(name="momentum", cxx_type="double", cxx_default=0.1),
        Arg(name="eps", cxx_type="double", cxx_default=1e-5),
    ],
    "nn.h::batch_norm_train_dxwb": [
        Arg(name="dy", cxx_type="value::BaseTensorValue"),
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="w", cxx_type="value::BaseTensorValue"),
        Arg(name="b", cxx_type="value::BaseTensorValue"),
        Arg(name="eps", cxx_type="double"),
    ],
    "nn.h::bias_add": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="bias", cxx_type="value::BaseTensorValue"),
        Arg(name="axis", cxx_type="int64_t", cxx_default=1),
    ],
    "nn.h::dropout": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="p", cxx_type="double", cxx_default=0.5),
        Arg(name="in_states", cxx_type=OptionalTensor, cxx_default="nullptr"),
    ],
    "nn.h::dropout_dx": [
        Arg(name="dy", cxx_type="value::BaseTensorValue"),
        Arg(name="mask", cxx_type="value::BaseTensorValue"),
        Arg(name="reserve_space", cxx_type="value::BaseTensorValue"),
        Arg(name="p", cxx_type="double", cxx_default=0.5),
    ],
    "nn.h::local_response_norm": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="size", cxx_type="int64_t"),
        Arg(name="alpha", cxx_type="double", cxx_default=1e-4),
        Arg(name="beta", cxx_type="double", cxx_default=0.75),
        Arg(name="k", cxx_type="double", cxx_default=1.0),
    ],
    "nn.h::conv_dxw": [
        Arg(name="x_or_w", cxx_type="value::BaseTensorValue"),
        Arg(name="y", cxx_type=OptionalTensor),
        Arg(name="dy", cxx_type="value::BaseTensorValue"),
        Arg(name="shape", cxx_type="value::Value"),
        Arg(name="stride", cxx_type="std::vector<int64_t>", cxx_normalizer="IntTuple"),
        Arg(name="padding", cxx_type="std::vector<int64_t>", cxx_normalizer="IntTuple"),
        Arg(name="dilation", cxx_type="std::vector<int64_t>", cxx_normalizer="IntTuple"),
        Arg(name="groups", cxx_type="int64_t"),
    ],
    "nn.h::conv_transpose_dxw": [
        Arg(name="x_or_w", cxx_type="value::BaseTensorValue"),
        Arg(name="y", cxx_type=OptionalTensor),
        Arg(name="dy", cxx_type="value::BaseTensorValue"),
        Arg(name="shape", cxx_type="value::Value"),
        Arg(name="stride", cxx_type="std::vector<int64_t>", cxx_normalizer="IntTuple"),
        Arg(name="padding", cxx_type="std::vector<int64_t>", cxx_normalizer="IntTuple"),
        Arg(name="output_padding", cxx_type="std::vector<int64_t>", cxx_normalizer="IntTuple"),
        Arg(name="dilation", cxx_type="std::vector<int64_t>", cxx_normalizer="IntTuple"),
        Arg(name="groups", cxx_type="int64_t"),
    ],
    "nn.h::pool_dx": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="y", cxx_type="value::BaseTensorValue"),
        Arg(name="dy", cxx_type="value::BaseTensorValue"),
        Arg(name="kernel", cxx_type="std::vector<int64_t>", cxx_normalizer="IntTuple"),
        Arg(name="stride", cxx_type="std::vector<int64_t>", cxx_normalizer="IntTuple"),
        Arg(name="padding", cxx_type="std::vector<int64_t>", cxx_normalizer="IntTuple"),
        Arg(name="dilation", cxx_type="std::vector<int64_t>", cxx_normalizer="IntTuple"),
        Arg(name="ceil_mode", cxx_type="bool"),
        Arg(name="include_pad", cxx_type="bool"),
    ],
    "nn.h::adaptive_pool_dx": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="y", cxx_type="value::BaseTensorValue"),
        Arg(name="dy", cxx_type="value::BaseTensorValue"),
        Arg(name="shape", cxx_type="std::vector<int64_t>", cxx_normalizer="IntTuple"),
    ],
    "nn.h::softmax_dx": [
        Arg(name="y", cxx_type="value::BaseTensorValue"),
        Arg(name="dy", cxx_type="value::BaseTensorValue"),
        Arg(name="axis", cxx_type="int", cxx_default=-1),
    ],
    "nn.h::take": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="indices", cxx_type="value::BaseTensorValue"),
        Arg(name="axis", cxx_type="value::Value", cxx_default="nullptr"),
        Arg(name="mode", cxx_type="std::string", cxx_default='"clip"', py_default='"clip"'),
    ],
    "nn.h::take_dx": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="dy", cxx_type="value::BaseTensorValue"),
        Arg(name="indices", cxx_type="value::BaseTensorValue"),
        Arg(name="axis", cxx_type="value::Value", cxx_default="nullptr"),
        Arg(name="mode", cxx_type="std::string", cxx_default='"clip"', py_default='"clip"'),
    ],
    "nn.h::embedding": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="indices", cxx_type="value::BaseTensorValue"),
    ],
    "nn.h::embedding_dx": [
        Arg(name="dy", cxx_type="value::BaseTensorValue"),
        Arg(name="indices", cxx_type="value::BaseTensorValue"),
        Arg(name="num_weight", cxx_type="value::Value"),
    ],
    "nn.h::repeat": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="repeats", cxx_type="int"),
        Arg(name="axis", cxx_type="value::Value", cxx_default="nullptr"),
    ],
    "nn.h::repeat_dx": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="dy", cxx_type="value::BaseTensorValue"),
        Arg(name="repeats", cxx_type="int"),
        Arg(name="axis", cxx_type="value::Value", cxx_default="nullptr"),
    ],
    "nn.h::sequence_mask": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="sequence_length", cxx_type="value::BaseTensorValue"),
        Arg(name="mask_value", cxx_type="double", cxx_default=0.0),
        Arg(name="axis", cxx_type="int", cxx_default=0),
    ],
    "nn.h::reverse_sequence": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="sequence_length", cxx_type="value::BaseTensorValue"),
        Arg(name="seq_axis", cxx_type="int", cxx_default=1),
        Arg(name="batch_axis", cxx_type="int", cxx_default=0),
    ],
    "nn.h::concatenate": [
        Arg(name="x", cxx_type="std::vector<value::BaseTensorValue>", cxx_normalizer="TensorTuple"),
        Arg(name="axis", cxx_type="int", cxx_default=0),
    ],
    "nn.h::mesh_grid": [
        Arg(name="x", cxx_type="std::vector<value::BaseTensorValue>", cxx_normalizer="TensorTuple"),
    ],
    "nn.h::stack": [
        Arg(name="x", cxx_type="std::vector<value::BaseTensorValue>", cxx_normalizer="TensorTuple"),
        Arg(name="axis", cxx_type="int", cxx_default=0),
    ],
    "nn.h::squeeze": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(
            name="axis",
            cxx_type="std::vector<int64_t>",
            cxx_normalizer="IntTuple",
            cxx_default="{}",
            py_default="None",
        ),
    ],
    "nn.h::layer_norm": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="scale", cxx_type=OptionalTensor, cxx_default="nullptr"),
        Arg(name="bias", cxx_type=OptionalTensor, cxx_default="nullptr"),
        Arg(name="axis", cxx_type="int64_t", cxx_default=-1),
        Arg(name="eps", cxx_type="double", cxx_default=1e-5),
    ],
    "nn.h::layer_norm_dx": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="scale", cxx_type=OptionalTensor),
        Arg(name="dy", cxx_type="value::BaseTensorValue"),
        Arg(name="axis", cxx_type="int64_t", cxx_default=-1),
        Arg(name="eps", cxx_type="double", cxx_default=1e-5),
    ],
    "nn.h::layer_norm_train_dx": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="scale", cxx_type=OptionalTensor),
        Arg(name="dy", cxx_type="value::BaseTensorValue"),
        Arg(name="mean", cxx_type="value::BaseTensorValue"),
        Arg(name="invvar", cxx_type="value::BaseTensorValue"),
        Arg(name="axis", cxx_type="int64_t", cxx_default=-1),
        Arg(name="eps", cxx_type="double", cxx_default=1e-5),
    ],
    "nn.h::split": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="indices_or_sections", cxx_type="value::Value", cxx_default="nullptr"),
        Arg(name="axis", cxx_type="int", cxx_default=0),
    ],
    "nn.h::threshold": [
        Arg(name="x", cxx_type="value::Value"),
        Arg(name="threshold", cxx_type="double", cxx_default=0.0),
        Arg(name="value", cxx_type="double", cxx_default=0.0),
    ],
    "nn.h::threshold_dx": [
        Arg(name="x", cxx_type="value::Value"),
        Arg(name="dy", cxx_type="value::BaseTensorValue"),
        Arg(name="threshold", cxx_type="double", cxx_default=0.0),
    ],
    "loss.h::loss": [
        Arg(name="y_true", cxx_type="value::BaseTensorValue"),
        Arg(name="y_pred", cxx_type="value::BaseTensorValue"),
    ],
    "loss.h::loss_dtp": [
        Arg(name="dy", cxx_type="value::BaseTensorValue"),
        Arg(name="y_true", cxx_type="value::BaseTensorValue"),
        Arg(name="y_pred", cxx_type="value::BaseTensorValue"),
    ],
    "ufunc.h::unary_ufunc": [
        Arg(name="x", cxx_type="value::Value"),
        Arg(name="out", cxx_type="value::Value", cxx_default="nullptr"),
        Arg(name="where", cxx_type="value::Value", cxx_default="nullptr"),
    ],
    "ufunc.h::binary_ufunc": [
        Arg(name="x1", cxx_type="value::Value"),
        Arg(name="x2", cxx_type="value::Value"),
        Arg(name="out", cxx_type="value::Value", cxx_default="nullptr"),
        Arg(name="where", cxx_type="value::Value", cxx_default="nullptr"),
    ],
    "ufunc.h::ternary_ufunc": [
        Arg(name="x1", cxx_type="value::Value"),
        Arg(name="x2", cxx_type="value::Value"),
        Arg(name="x3", cxx_type="value::Value"),
        Arg(name="out", cxx_type="value::Value", cxx_default="nullptr"),
        Arg(name="where", cxx_type="value::Value", cxx_default="nullptr"),
    ],
    "ufunc.h::unary": [
        Arg(name="x", cxx_type="value::Value"),
    ],
    "ufunc.h::binary": [
        Arg(name="x1", cxx_type="value::Value"),
        Arg(name="x2", cxx_type="value::Value"),
    ],
    "ufunc.h::ternary": [
        Arg(name="x1", cxx_type="value::Value"),
        Arg(name="x2", cxx_type="value::Value"),
        Arg(name="x3", cxx_type="value::Value"),
    ],
    "ufunc.h::unary_dx": [
        Arg(name="x", cxx_type=OptionalValue),
        Arg(name="y", cxx_type=OptionalTensor),
        Arg(name="dy", cxx_type="value::BaseTensorValue"),
    ],
    "ufunc.h::binary_dx": [
        Arg(name="x1", cxx_type="value::Value"),
        Arg(name="x2", cxx_type="value::Value"),
        Arg(name="y", cxx_type="value::BaseTensorValue"),
        Arg(name="dy", cxx_type="value::BaseTensorValue"),
    ],
    "ufunc.h::ternary_dx": [
        Arg(name="x1", cxx_type="value::Value"),
        Arg(name="x2", cxx_type="value::Value"),
        Arg(name="x3", cxx_type="value::Value"),
        Arg(name="y", cxx_type="value::BaseTensorValue"),
        Arg(name="dy", cxx_type="value::BaseTensorValue"),
    ],
    "likes.h::binary_to": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="shape", cxx_type="std::vector<int64_t>", cxx_normalizer="IntTuple"),
    ],
    "likes.h::binary_like": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="like_type", cxx_type="value::BaseTensorValue"),
    ],
    "likes.h::reshape": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="shape", cxx_type="value::Value"),
        Arg(name="reverse", cxx_type="bool", cxx_default=False),
    ],
    "likes.h::resize2d": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="size", cxx_type="value::Value"),
        Arg(name="layout", cxx_type="std::string", cxx_default='"NCHW"', py_default='"NCHW"'),
        Arg(name="method", cxx_type="std::string", cxx_default='"linear"', py_default='"linear"'),
        Arg(
            name="coordinate_transformation_mode",
            cxx_type="std::string",
            cxx_default='"half_pixel"',
            py_default='"half_pixel"',
        ),
        Arg(name="rounding_method", cxx_type="std::string", cxx_default="", py_default='""'),
        Arg(name="cubic_alpha", cxx_type="float", cxx_default="-0.5", py_default="-0.5"),
        Arg(name="cubic_exclude", cxx_type="int", cxx_default="0", py_default="0"),
        Arg(name="out_dtype", cxx_type="std::string", cxx_default="", py_default='""'),
    ],
    "likes.h::resize2d_dx": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="dy", cxx_type="value::BaseTensorValue"),
        Arg(name="size", cxx_type="std::vector<int64_t>", cxx_normalizer="IntTuple"),
        Arg(name="layout", cxx_type="std::string", cxx_default='"NCHW"', py_default='"NCHW"'),
        Arg(name="method", cxx_type="std::string", cxx_default='"linear"', py_default='"linear"'),
        Arg(
            name="coordinate_transformation_mode",
            cxx_type="std::string",
            cxx_default='"half_pixel"',
            py_default='"half_pixel"',
        ),
        Arg(name="rounding_method", cxx_type="std::string", cxx_default="", py_default='""'),
        Arg(name="cubic_alpha", cxx_type="float", cxx_default="-0.5", py_default="-0.5"),
        Arg(name="cubic_exclude", cxx_type="int", cxx_default="0", py_default="0"),
        Arg(name="out_dtype", cxx_type="std::string", cxx_default="", py_default='""'),
    ],
    "reduce.h::reduce": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(
            name="axis",
            cxx_type="std::vector<int64_t>",
            cxx_normalizer="IntTuple",
            cxx_default="{}",
            py_default=(),
        ),
        Arg(name="keepdims", cxx_type="bool", cxx_default=False),
        Arg(name="exclude", cxx_type="bool", cxx_default=False),
    ],
    "reduce.h::l2norm": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
    ],
    "reduce.h::prod_dx": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="dy", cxx_type="value::BaseTensorValue"),
        Arg(
            name="axis",
            cxx_type="std::vector<int64_t>",
            cxx_normalizer="IntTuple",
            cxx_default="{}",
            py_default=(),
        ),
        Arg(name="keepdims", cxx_type="bool", cxx_default=False),
        Arg(name="exclude", cxx_type="bool", cxx_default=False),
    ],
    "reduce.h::mean_dx": [
        Arg(name="dy", cxx_type="value::BaseTensorValue"),
        Arg(name="shape", cxx_type="value::Value"),
        Arg(
            name="axis",
            cxx_type="std::vector<int64_t>",
            cxx_normalizer="IntTuple",
            cxx_default="{}",
            py_default=(),
        ),
        Arg(name="keepdims", cxx_type="bool", cxx_default=False),
        Arg(name="exclude", cxx_type="bool", cxx_default=False),
    ],
    "transform.h::arange": [
        Arg(name="start", cxx_type="value::BaseTensorValue"),
        Arg(name="stop", cxx_type="value::BaseTensorValue"),
        Arg(name="step", cxx_type="value::BaseTensorValue"),
        Arg(name="dtype", cxx_type="std::string", cxx_default='"float32"', py_default='"float32"'),
        Arg(
            name="device",
            cxx_type="std::string",
            cxx_default='"cpu"',
            py_default='"cpu"',
        ),
    ],
    "transform.h::adv_index": [
        Arg(
            name="inputs",
            cxx_type="std::vector<value::BaseTensorValue>",
            cxx_normalizer="TensorTuple",
        ),
    ],
    "transform.h::adv_index_dx": [
        Arg(name="dy", cxx_type="value::BaseTensorValue"),
        Arg(
            name="inputs",
            cxx_type="std::vector<value::BaseTensorValue>",
            cxx_normalizer="TensorTuple",
        ),
    ],
    "transform.h::scatter": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="index", cxx_type="value::BaseTensorValue"),
        Arg(name="src", cxx_type="value::BaseTensorValue"),
        Arg(name="axis", cxx_type="value::Value"),
    ],
    "transform.h::scatter_dx": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="y", cxx_type="value::BaseTensorValue"),
        Arg(name="dy", cxx_type="value::BaseTensorValue"),
        Arg(name="index", cxx_type="value::BaseTensorValue"),
        Arg(name="src", cxx_type="value::BaseTensorValue"),
        Arg(name="axis", cxx_type="value::Value"),
    ],
    "transform.h::transpose": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(
            name="axes",
            cxx_type="std::vector<int64_t>",
            cxx_normalizer="IntTuple",
            cxx_default="{}",
            py_default="None",
        ),
    ],
    "transform.h::swap_axis": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="axis1", cxx_type="int"),
        Arg(name="axis2", cxx_type="int"),
    ],
    "transform.h::reverse": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="axis", cxx_type="int", cxx_default=0),
    ],
    "transform.h::clip": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="a_min", cxx_type="double"),
        Arg(name="a_max", cxx_type="double"),
    ],
    "transform.h::clip_dx": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="dy", cxx_type="value::BaseTensorValue"),
        Arg(name="a_min", cxx_type="double"),
        Arg(name="a_max", cxx_type="double"),
    ],
    "transform.h::cast": [
        Arg(name="data", cxx_type="value::BaseTensorValue"),
        Arg(name="dtype", cxx_type="std::string"),
    ],
    "transform.h::group_cast": [
        Arg(
            name="tensor_list",
            cxx_type="std::vector<value::BaseTensorValue>",
            cxx_normalizer="TensorTuple",
        ),
        Arg(name="dtype", cxx_type="std::string"),
    ],
    "transform.h::strided_slice": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="begin", cxx_type="value::Value"),
        Arg(name="end", cxx_type="value::Value"),
        Arg(
            name="strides",
            cxx_type="std::vector<int64_t>",
            cxx_normalizer="IntTuple",
            cxx_default="{}",
            py_default="None",
        ),
        Arg(name="slice_mode", cxx_type="std::string", cxx_default='"end"', py_default='"end"'),
    ],
    "transform.h::strided_set": [
        Arg(name="data", cxx_type="value::BaseTensorValue"),
        Arg(name="v", cxx_type="value::BaseTensorValue"),
        Arg(name="begin", cxx_type="std::vector<int64_t>", cxx_normalizer="IntTuple"),
        Arg(name="end", cxx_type="std::vector<int64_t>", cxx_normalizer="IntTuple"),
        Arg(
            name="strides",
            cxx_type="std::vector<int64_t>",
            cxx_normalizer="IntTuple",
            cxx_default="{}",
            py_default="None",
        ),
    ],
    "likes.h::sum_dx": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="dy", cxx_type="value::BaseTensorValue"),
        Arg(
            name="axis",
            cxx_type="std::vector<int64_t>",
            cxx_normalizer="IntTuple",
            cxx_default="{}",
            py_default=(),
        ),
        Arg(
            name="keepdims",
            cxx_type="std::vector<int64_t>",
            cxx_normalizer="IntTuple",
            cxx_default="{0}",
            py_default=(0),
        ),
        Arg(name="exclude", cxx_type="bool", cxx_default=False),
    ],
    "likes.h::sum": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(
            name="axis",
            cxx_type="std::vector<int64_t>",
            cxx_normalizer="IntTuple",
            cxx_default="{}",
            py_default=(),
        ),
        Arg(
            name="keepdims",
            cxx_type="std::vector<int64_t>",
            cxx_normalizer="IntTuple",
            cxx_default="{0}",
            py_default=(0),
        ),
        Arg(name="exclude", cxx_type="bool", cxx_default=False),
    ],
    "transform.h::cumsum": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="axis", cxx_type="int"),
        Arg(name="dtype", cxx_type="std::string", cxx_default='"float32"', py_default='"float32"'),
        Arg(name="exclusive", cxx_type="bool", cxx_default=False),
    ],
    "transform.h::strided_slice_dx": [
        Arg(name="dy", cxx_type="value::BaseTensorValue"),
        Arg(name="shape", cxx_type="value::Value"),
        Arg(name="begin", cxx_type="std::vector<int64_t>", cxx_normalizer="IntTuple"),
        Arg(name="end", cxx_type="std::vector<int64_t>", cxx_normalizer="IntTuple"),
        Arg(
            name="strides",
            cxx_type="std::vector<int64_t>",
            cxx_normalizer="IntTuple",
            cxx_default="{}",
            py_default="None",
        ),
        Arg(name="slice_mode", cxx_type="std::string", cxx_default='"end"', py_default='"end"'),
    ],
    "transform.h::expand_dims": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="axis", cxx_type="int"),
        Arg(name="num_newaxis", cxx_type="int", cxx_default=1),
    ],
    "random.h::threefry_generate": [
        Arg(name="key", cxx_type="value::BaseTensorValue"),
        Arg(name="shape", cxx_type="std::vector<int64_t>", cxx_normalizer="IntTuple"),
    ],
    "random.h::threefry_split": [
        Arg(name="key", cxx_type="value::BaseTensorValue"),
    ],
    "vision.h::get_valid_counts": [
        Arg(name="data", cxx_type="value::BaseTensorValue"),
        Arg(name="score_threshold", cxx_type="value::BaseTensorValue"),
        Arg(name="id_index", cxx_type="int64_t", cxx_default=0),
        Arg(name="score_index", cxx_type="int64_t", cxx_default=1),
    ],
    "vision.h::non_max_suppression": [
        Arg(name="data", cxx_type="value::BaseTensorValue"),
        Arg(name="valid_count", cxx_type="value::BaseTensorValue"),
        Arg(name="indices", cxx_type="value::BaseTensorValue"),
        Arg(name="max_output_size", cxx_type="value::BaseTensorValue"),
        Arg(name="iou_threshold", cxx_type="value::BaseTensorValue"),
        Arg(name="force_suppress", cxx_type="bool", cxx_default=False),
        Arg(name="top_k", cxx_type="int64_t", cxx_default=-1),
        Arg(name="coord_start", cxx_type="int64_t", cxx_default=2),
        Arg(name="score_index", cxx_type="int64_t", cxx_default=1),
        Arg(name="id_index", cxx_type="int64_t", cxx_default=0),
        Arg(name="return_indices", cxx_type="bool", cxx_default=True),
        Arg(name="invalid_to_bottom", cxx_type="bool", cxx_default=False),
    ],
    "vision.h::roi_align": [
        Arg(name="data", cxx_type="value::BaseTensorValue"),
        Arg(name="rois", cxx_type="value::BaseTensorValue"),
        Arg(name="pooled_size", cxx_type="std::vector<int64_t>", cxx_normalizer="IntTuple"),
        Arg(name="spatial_scale", cxx_type="double"),
        Arg(name="sample_ratio", cxx_type="int64_t", cxx_default=-1, py_default=-1),
        Arg(name="layout", cxx_type="std::string", cxx_default='"NCHW"', py_default='"NCHW"'),
        Arg(name="mode", cxx_type="std::string", cxx_default='"avg"', py_default='"avg"'),
    ],
    "vision.h::roi_align_dx": [
        Arg(name="data", cxx_type="value::BaseTensorValue"),
        Arg(name="rois", cxx_type="value::BaseTensorValue"),
        Arg(name="dy", cxx_type="value::BaseTensorValue"),
        Arg(name="pooled_size", cxx_type="std::vector<int64_t>", cxx_normalizer="IntTuple"),
        Arg(name="spatial_scale", cxx_type="double"),
        Arg(name="sample_ratio", cxx_type="int64_t", cxx_default=-1, py_default=-1),
        Arg(name="layout", cxx_type="std::string", cxx_default='"NCHW"', py_default='"NCHW"'),
        Arg(name="mode", cxx_type="std::string", cxx_default='"avg"', py_default='"avg"'),
    ],
    "optimizer.h::sgd": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="dx", cxx_type="value::BaseTensorValue"),
        Arg(name="v", cxx_type="value::BaseTensorValue"),
        Arg(name="learning_rate", cxx_type="double"),
        Arg(name="mu", cxx_type="double"),
    ],
    "optimizer.h::lans": [
        Arg(
            name="tensor_list",
            cxx_type="std::vector<value::BaseTensorValue>",
            cxx_normalizer="TensorTuple",
        ),
        Arg(name="step", cxx_type="value::BaseTensorValue"),
        Arg(name="learning_rate", cxx_type="float"),
        Arg(name="beta1", cxx_type="float"),
        Arg(name="beta2", cxx_type="float"),
        Arg(name="eps", cxx_type="float"),
        Arg(name="bias_correction", cxx_type="int"),
        Arg(name="weight_decay", cxx_type="float"),
        Arg(name="grad_averaging", cxx_type="int"),
        Arg(name="mode", cxx_type="int"),
        Arg(name="normalize_grad", cxx_type="bool"),
    ],
    "stream.h::stream": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="stream_tag", cxx_type="int", cxx_default=0),
    ],
    "communication.h::allreduce": [
        Arg(name="x", cxx_type="std::vector<value::BaseTensorValue>", cxx_normalizer="TensorTuple"),
        Arg(name="computation", cxx_type="std::string", cxx_default='"sum"', py_default='"sum"'),
        Arg(name="rank_list", cxx_type="value::Value", cxx_default="nullptr"),
    ],
    "communication.h::comm_reduce": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="root", cxx_type="int"),
        Arg(name="computation", cxx_type="std::string", cxx_default='"sum"', py_default='"sum"'),
        Arg(name="rank_list", cxx_type="value::Value", cxx_default="nullptr"),
    ],
    "communication.h::allgather": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="axis", cxx_type="int"),
        Arg(name="rank_list", cxx_type="value::Value", cxx_default="nullptr"),
    ],
    "communication.h::group_allgather": [
        Arg(
            name="tensor_list",
            cxx_type="std::vector<value::BaseTensorValue>",
            cxx_normalizer="TensorTuple",
        ),
        Arg(name="axis", cxx_type="int"),
        Arg(
            name="out", cxx_type="std::vector<value::BaseTensorValue>", cxx_normalizer="TensorTuple"
        ),
    ],
    "communication.h::reduce_scatter": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="computation", cxx_type="std::string", cxx_default='"sum"', py_default='"sum"'),
        Arg(name="rank_list", cxx_type="value::Value", cxx_default="nullptr"),
    ],
    "communication.h::group_reduce_scatter": [
        Arg(
            name="tensor_list",
            cxx_type="std::vector<value::BaseTensorValue>",
            cxx_normalizer="TensorTuple",
        ),
        Arg(name="computation", cxx_type="std::string", cxx_default='"sum"', py_default='"sum"'),
    ],
    "communication.h::broadcast": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="root", cxx_type="int"),
        Arg(name="rank_list", cxx_type="value::Value", cxx_default="nullptr"),
    ],
    "communication.h::all_to_all": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="rank_list", cxx_type="value::Value", cxx_default="nullptr"),
    ],
    "communication.h::send": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="peer", cxx_type="int"),
        Arg(name="token", cxx_type=OptionalTensor, cxx_default="nullptr", py_default="None"),
    ],
    "communication.h::recv": [
        Arg(name="peer", cxx_type="int"),
        Arg(name="shape", cxx_type="std::vector<int64_t>", cxx_normalizer="IntTuple"),
        Arg(name="dtype", cxx_type="std::string", cxx_default='"float32"', py_default='"float32"'),
        Arg(name="token", cxx_type=OptionalTensor, cxx_default="nullptr", py_default="None"),
    ],
    "communication.h::gather_scatter": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="root", cxx_type="int"),
    ],
    "transform.h::gather": [
        Arg(name="data", cxx_type="value::BaseTensorValue"),
        Arg(name="axis", cxx_type="int"),
        Arg(name="indices", cxx_type="value::BaseTensorValue"),
    ],
    "transform.h::gather_dx": [
        Arg(name="data", cxx_type="value::BaseTensorValue"),
        Arg(name="axis", cxx_type="int"),
        Arg(name="indices", cxx_type="value::BaseTensorValue"),
        Arg(name="dy", cxx_type="value::BaseTensorValue"),
    ],
    "transform.h::gather_nd": [
        Arg(name="data", cxx_type="value::BaseTensorValue"),
        Arg(name="indices", cxx_type="value::BaseTensorValue"),
    ],
    "transform.h::gather_nd_dx": [
        Arg(name="data", cxx_type="value::BaseTensorValue"),
        Arg(name="indices", cxx_type="value::BaseTensorValue"),
        Arg(name="dy", cxx_type="value::BaseTensorValue"),
    ],
    "algorithm.h::argsort": [
        Arg(name="data", cxx_type="value::BaseTensorValue"),
        Arg(name="axis", cxx_type="int", cxx_default=-1),
        Arg(name="is_ascend", cxx_type="bool", cxx_default=True),
        Arg(name="dtype", cxx_type="std::string", cxx_default='"int"', py_default='"int32"'),
    ],
    "algorithm.h::sort": [
        Arg(name="data", cxx_type="value::BaseTensorValue"),
        Arg(name="axis", cxx_type="int", cxx_default=-1),
        Arg(name="is_ascend", cxx_type="bool", cxx_default=True),
    ],
    "transform.h::full": [
        Arg(name="fill_value", cxx_type="double"),
        Arg(name="shape", cxx_type="value::Value"),
        Arg(name="dtype", cxx_type="std::string", cxx_default='"int"', py_default='"int32"'),
        Arg(
            name="device",
            cxx_type="std::string",
            cxx_default='"cpu"',
            py_default='"cpu"',
        ),
    ],
    "transform.h::full_like": [
        Arg(name="data", cxx_type="value::BaseTensorValue"),
        Arg(name="fill_value", cxx_type="double"),
    ],
    "transform.h::where": [
        Arg(name="condition", cxx_type="value::BaseTensorValue"),
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="y", cxx_type="value::BaseTensorValue"),
    ],
    "memory.h::device_copy": [
        Arg(name="data", cxx_type="value::BaseTensorValue"),
        Arg(
            name="src_device",
            cxx_type="std::string",
            cxx_default='"cpu"',
            py_default='"cpu"',
        ),
        Arg(
            name="dst_device",
            cxx_type="std::string",
            cxx_default='"cpu"',
            py_default='"cpu"',
        ),
    ],
    "memory.h::fuse_tensor": [
        Arg(
            name="data",
            cxx_type="std::vector<value::BaseTensorValue>",
            cxx_normalizer="TensorTuple",
        ),
    ],
    "memory.h::defuse_tensor": [
        Arg(name="data", cxx_type="value::BaseTensorValue"),
        Arg(name="sizes", cxx_type="std::vector<int64_t>", cxx_normalizer="IntTuple"),
        Arg(name="shapes", cxx_type="std::vector<int64_t>", cxx_normalizer="IntTuple"),
        Arg(name="shape_indices", cxx_type="std::vector<int64_t>", cxx_normalizer="IntTuple"),
    ],
    "algorithm.h::topk": [
        Arg(name="data", cxx_type="value::BaseTensorValue"),
        Arg(name="k", cxx_type="value::Value"),
        Arg(name="axis", cxx_type="int", cxx_default=-1),
        Arg(name="ret_type", cxx_type="std::string", cxx_default='"both"', py_default='"both"'),
        Arg(name="is_ascend", cxx_type="bool", cxx_default=False),
        Arg(name="dtype", cxx_type="std::string", cxx_default='"int64_t"', py_default='"int64"'),
    ],
    "init.h::init_op": [
        Arg(name="shape", cxx_type="value::Value"),
        Arg(name="dtype", cxx_type="std::string", cxx_default='"int"', py_default='"int32"'),
        Arg(
            name="device",
            cxx_type="std::string",
            cxx_default='"cpu"',
            py_default='"cpu"',
        ),
    ],
    "init.h::one_hot": [
        Arg(name="indices", cxx_type="value::BaseTensorValue"),
        Arg(name="on_value", cxx_type="value::BaseTensorValue"),
        Arg(name="off_value", cxx_type="value::BaseTensorValue"),
        Arg(name="depth", cxx_type="int64_t"),
        Arg(name="axis", cxx_type="int64_t", cxx_default=-1),
        Arg(name="dtype", cxx_type="std::string", cxx_default='"int"', py_default='"int32"'),
        Arg(
            name="device",
            cxx_type="std::string",
            cxx_default='"cpu"',
            py_default='"cpu"',
        ),
    ],
    "vm.h::alloc_storage": [
        Arg(name="size", cxx_type="value::Value"),
        Arg(name="alignment", cxx_type="value::Value"),
        Arg(name="device_type", cxx_type="int"),
        Arg(name="device_id", cxx_type="int"),
        Arg(name="dtype", cxx_type="std::string", cxx_default='"float32"', py_default='"float32"'),
        Arg(name="alloc_async", cxx_type="bool", cxx_default=True),
    ],
    "vm.h::alloc_tensor": [
        Arg(name="storage", cxx_type="value::BaseTensorValue"),
        Arg(name="shape", cxx_type="value::Value"),
        Arg(name="dtype", cxx_type="std::string", cxx_default='"float32"', py_default='"float32"'),
        Arg(
            name="assert_shape",
            cxx_type="std::vector<int64_t>",
            cxx_normalizer="IntTuple",
            cxx_default="{}",
            py_default="None",
        ),
        Arg(name="own", cxx_type="bool", cxx_default=True),
    ],
    "vm.h::free": [
        Arg(name="memory", cxx_type="value::BaseTensorValue"),
    ],
    "vm.h::invoke_op": [
        Arg(name="func", cxx_type="value::Value"),
        Arg(name="inputs", cxx_type="value::Value"),
        Arg(name="outputs", cxx_type="value::Value"),
    ],
    "vm.h::infer_type": [
        Arg(name="func", cxx_type="value::Value"),
        Arg(name="inputs", cxx_type="value::Value"),
    ],
    "vm.h::set_shape": [
        Arg(name="data", cxx_type="value::BaseTensorValue"),
        Arg(name="shape", cxx_type="value::Value"),
    ],
    "transform.h::argwhere": [
        Arg(name="condition", cxx_type="value::BaseTensorValue"),
    ],
    "stream.h::set_stream": [
        Arg(name="device_id", cxx_type="int64_t"),
        Arg(name="stream_id", cxx_type="int64_t"),
    ],
    "stream.h::event": [
        Arg(name="event_id", cxx_type="int64_t"),
        Arg(name="stream_id", cxx_type="int64_t", cxx_default=-1, py_default=-1),
    ],
    "stream.h::stream_barrier": [
        # empty arg list
    ],
    "transform.h::size": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="axis", cxx_type="value::Value", cxx_default="nullptr"),
    ],
}


def _sanity_check(name, schema):
    is_optional = False

    for arg in schema:
        if arg.cxx_default is None and is_optional:
            raise ValueError(f"In {name}, required arguments should precede optional arguments")

        if arg.cxx_default is not None:
            is_optional = True


def by_file():
    files = defaultdict(dict)
    existing = set()

    for name, schema in SCHEMAS.items():
        _sanity_check(name, schema)
        file_name, schema_name = name.split("::")
        assert schema_name not in existing
        existing.add(schema_name)
        files[file_name][schema_name] = schema

    return files


def by_name():
    schemas = dict()

    for name, schema in SCHEMAS.items():
        _sanity_check(name, schema)
        _, schema_name = name.split("::")
        assert schema_name not in schemas
        schemas[schema_name] = schema

    return schemas
