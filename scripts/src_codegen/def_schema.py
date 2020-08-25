from collections import defaultdict

from .codegen_utils import Arg

SCHEMAS = {
    "nn.h::conv": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="w", cxx_type="value::BaseTensorValue"),
        Arg(name="stride",
            cxx_type="std::vector<int64_t>",
            cxx_default="{1}",
            py_default=1,
            cxx_normalizer="IntTuple"),
        Arg(name="padding",
            cxx_type="std::vector<int64_t>",
            cxx_default="{0}",
            py_default=0,
            cxx_normalizer="IntTuple"),
        Arg(name="dilation",
            cxx_type="std::vector<int64_t>",
            cxx_default="{1}",
            py_default=1,
            cxx_normalizer="IntTuple"),
        Arg(name="groups", cxx_type="int64_t", cxx_default=1),
    ],
    "nn.h::pool": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="kernel",
            cxx_type="std::vector<int64_t>",
            cxx_normalizer="IntTuple"),
        Arg(name="stride",
            cxx_type="std::vector<int64_t>",
            cxx_normalizer="IntTuple"),
        Arg(name="padding",
            cxx_type="std::vector<int64_t>",
            cxx_default="{0}",
            py_default=0,
            cxx_normalizer="IntTuple"),
        Arg(name="dilation",
            cxx_type="std::vector<int64_t>",
            cxx_default="{1}",
            py_default=1,
            cxx_normalizer="IntTuple"),
        Arg(name="ceil_mode", cxx_type="bool", cxx_default=False),
        Arg(name="include_pad", cxx_type="bool", cxx_default=True),
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
    "nn.h::local_response_norm": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="size", cxx_type="int64_t"),
        Arg(name="alpha", cxx_type="double", cxx_default=1e-4),
        Arg(name="beta", cxx_type="double", cxx_default=0.75),
        Arg(name="k", cxx_type="double", cxx_default=1.0),
    ],
    "nn.h::conv_dxw": [
        Arg(name="x_or_w", cxx_type="value::BaseTensorValue"),
        Arg(name="y", cxx_type="value::BaseTensorValue"),
        Arg(name="dy", cxx_type="value::BaseTensorValue"),
        Arg(name="shape", cxx_type="std::vector<int64_t>",
            cxx_normalizer="IntTuple"),
        Arg(name="stride", cxx_type="std::vector<int64_t>",
            cxx_normalizer="IntTuple"),
        Arg(name="padding",
            cxx_type="std::vector<int64_t>",
            cxx_normalizer="IntTuple"),
        Arg(name="dilation",
            cxx_type="std::vector<int64_t>",
            cxx_normalizer="IntTuple"),
        Arg(name="groups", cxx_type="int64_t"),
    ],
    "nn.h::pool_dx": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="y", cxx_type="value::BaseTensorValue"),
        Arg(name="dy", cxx_type="value::BaseTensorValue"),
        Arg(name="kernel",
            cxx_type="std::vector<int64_t>",
            cxx_normalizer="IntTuple"),
        Arg(name="stride",
            cxx_type="std::vector<int64_t>",
            cxx_normalizer="IntTuple"),
        Arg(name="padding",
            cxx_type="std::vector<int64_t>",
            cxx_normalizer="IntTuple"),
        Arg(name="dilation",
            cxx_type="std::vector<int64_t>",
            cxx_normalizer="IntTuple"),
        Arg(name="ceil_mode", cxx_type="bool"),
        Arg(name="include_pad", cxx_type="bool"),
    ],
    "nn.h::softmax_dx": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="y", cxx_type="value::BaseTensorValue"),
        Arg(name="dy", cxx_type="value::BaseTensorValue"),
        Arg(name="axis", cxx_type="int", cxx_default=-1),
    ],
    "nn.h::take": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="indices", cxx_type="value::BaseTensorValue"),
        Arg(name="axis", cxx_type="value::Value", cxx_default="nullptr"),
    ],
    "nn.h::take_dx": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="y", cxx_type="value::BaseTensorValue"),
        Arg(name="dy", cxx_type="value::BaseTensorValue"),
        Arg(name="indices", cxx_type="value::BaseTensorValue"),
        Arg(name="axis", cxx_type="value::Value", cxx_default="nullptr"),
    ],
    "nn.h::expand_dims": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="axis", cxx_type="int"),
        Arg(name="num_newaxis", cxx_type="int", cxx_default=1),
    ],
    "nn.h::repeat": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
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
    "nn.h::broadcast_to": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="shape",
            cxx_type="std::vector<int64_t>",
            cxx_normalizer="IntTuple"),
    ],
    "nn.h::broadcast_to_like": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="broadcast_type", cxx_type="value::BaseTensorValue"),
    ],
    "nn.h::concatenate": [
        Arg(name="x",
            cxx_type="std::vector<value::BaseTensorValue>",
            cxx_normalizer="TensorTuple"),
        Arg(name="axis", cxx_type="int", cxx_default=0),
    ],
    "nn.h::stack": [
        Arg(name="x",
            cxx_type="std::vector<value::BaseTensorValue>",
            cxx_normalizer="TensorTuple"),
        Arg(name="axis", cxx_type="int", cxx_default=0),
    ],
    "nn.h::layer_norm": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="axis", cxx_type="int64_t", cxx_default=-1),
        Arg(name="eps", cxx_type="double", cxx_default=1e-5),
    ],
    "nn.h::layer_norm_dx": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="y", cxx_type="value::BaseTensorValue"),
        Arg(name="dy", cxx_type="value::BaseTensorValue"),
        Arg(name="axis", cxx_type="int64_t", cxx_default=-1),
        Arg(name="eps", cxx_type="double", cxx_default=1e-5),
    ],
    "nn.h::split": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="indices_or_sections",
            cxx_type="std::vector<int64_t>",
            cxx_normalizer="IntTuple"),
        Arg(name="axis", cxx_type="int", cxx_default=0),
    ],
    "loss.h::loss": [
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
        Arg(name="x", cxx_type="value::Value"),
        Arg(name="y", cxx_type="value::BaseTensorValue"),
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
    "likes.h::collapse_like": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="shape",
            cxx_type="std::vector<int64_t>",
            cxx_normalizer="IntTuple"),
    ],
    "likes.h::reshape": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="shape",
            cxx_type="std::vector<int64_t>",
            cxx_normalizer="IntTuple"),
        Arg(name="reverse", cxx_type="bool", cxx_default=False),
    ],
    "likes.h::sum": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="axis",
            cxx_type="std::vector<int64_t>",
            cxx_normalizer="IntTuple"),
        Arg(name="keep",
            cxx_type="std::vector<int64_t>",
            cxx_normalizer="IntTuple"),
    ],
    "reduce.h::reduce": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="axis",
            cxx_type="std::vector<int64_t>",
            cxx_normalizer="IntTuple",
            cxx_default="{}",
            py_default=()),
        Arg(name="keepdims", cxx_type="bool", cxx_default=False),
    ],
    "reduce.h::reduce_dx": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="y", cxx_type="value::BaseTensorValue"),
        Arg(name="dy", cxx_type="value::BaseTensorValue"),
        Arg(name="axis",
            cxx_type="std::vector<int64_t>",
            cxx_normalizer="IntTuple",
            cxx_default="{}",
            py_default=()),
        Arg(name="keepdims", cxx_type="bool", cxx_default=False),
    ],
    "transform.h::transpose": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="axes",
            cxx_type="std::vector<int64_t>",
            cxx_normalizer="IntTuple",
            cxx_default="{}",
            py_default="None"),
    ],
    "transform.h::transpose_dx": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="y", cxx_type="value::BaseTensorValue"),
        Arg(name="dy", cxx_type="value::BaseTensorValue"),
        Arg(name="axes",
            cxx_type="std::vector<int64_t>",
            cxx_normalizer="IntTuple",
            cxx_default="{}",
            py_default="None"),
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
    "vision.h::get_valid_counts": [
        Arg(name="data", cxx_type="value::BaseTensorValue"),
        Arg(name="score_threshold", cxx_type="double",
            cxx_default=0),
        Arg(name="id_index", cxx_type="int64_t",
            cxx_default=0),
        Arg(name="score_index", cxx_type="int64_t",
            cxx_default=1),
    ],
    "vision.h::non_max_suppression": [
        Arg(name="data", cxx_type="value::BaseTensorValue"),
        Arg(name="valid_count", cxx_type="value::BaseTensorValue"),
        Arg(name="indices", cxx_type="value::BaseTensorValue"),
        Arg(name="max_output_size", cxx_type="value::BaseTensorValue"),
        Arg(name="iou_threshold", cxx_type="double",
            cxx_default=0.5),
        Arg(name="force_suppress", cxx_type="bool",
            cxx_default=False),
        Arg(name="top_k", cxx_type="int64_t",
            cxx_default=-1),
        Arg(name="coord_start", cxx_type="int64_t",
            cxx_default=2),
        Arg(name="score_index", cxx_type="int64_t",
            cxx_default=1),
        Arg(name="id_index", cxx_type="int64_t",
            cxx_default=0),
        Arg(name="return_indices", cxx_type="bool",
            cxx_default=True),
        Arg(name="invalid_to_bottom", cxx_type="bool",
            cxx_default=False),
    ],
    "optimizer.h::sgd": [
        Arg(name="x", cxx_type="value::BaseTensorValue"),
        Arg(name="dx", cxx_type="value::BaseTensorValue"),
        Arg(name="v", cxx_type="value::BaseTensorValue"),
        Arg(name="learning_rate", cxx_type="double"),
        Arg(name="mu", cxx_type="double"),
    ],
    "communication.h::_allreduce": [
        Arg(name="x",
            cxx_type="std::vector<value::BaseTensorValue>",
            cxx_normalizer="TensorTuple"),
    ]
}


def _sanity_check(name, schema):
    is_optional = False

    for arg in schema:
        if arg.cxx_default is None and is_optional:
            raise ValueError(
                f"In {name}, required arguments should precede optional arguments"
            )

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
