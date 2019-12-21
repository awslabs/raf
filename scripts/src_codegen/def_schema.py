from collections import defaultdict

from codegen_utils import Arg

SCHEMAS = {
    "nn.h::conv": [
        Arg(name="x", cxx_type="value::TensorValue"),
        Arg(name="w", cxx_type="value::TensorValue"),
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
        Arg(name="x", cxx_type="value::TensorValue"),
        Arg(name="kernel",
            cxx_type="std::vector<int64_t>",
            cxx_normalizer="IntTuple"),
        Arg(name="stride",
            cxx_type="std::vector<int64_t>",
            cxx_default="{}",
            py_default="None",
            cxx_normalizer="OptionalIntTuple"),
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
        Arg(name="x", cxx_type="value::TensorValue"),
        Arg(name="axis", cxx_type="int", cxx_default=-1),
    ],
    "nn.h::batch_norm": [
        Arg(name="x", cxx_type="value::TensorValue"),
        Arg(name="running_mean", cxx_type="value::TensorValue"),
        Arg(name="running_var", cxx_type="value::TensorValue"),
        Arg(name="w", cxx_type="value::TensorValue", cxx_default="nullptr"),
        Arg(name="b", cxx_type="value::TensorValue", cxx_default="nullptr"),
        Arg(name="momentum", cxx_type="double", cxx_default=0.1),
        Arg(name="eps", cxx_type="double", cxx_default=1e-5),
    ],
    "nn.h::batch_norm_train_dxwb": [
        Arg(name="dy", cxx_type="value::TensorValue"),
        Arg(name="x", cxx_type="value::TensorValue"),
        Arg(name="w", cxx_type="value::TensorValue"),
        Arg(name="b", cxx_type="value::TensorValue"),
        Arg(name="eps", cxx_type="double"),
    ],
    "gemm.h::matmul": [
        Arg(name="a", cxx_type="value::TensorValue"),
        Arg(name="b", cxx_type="value::TensorValue"),
        Arg(name="transpose_a", cxx_type="bool", cxx_default=False),
        Arg(name="transpose_b", cxx_type="bool", cxx_default=False),
    ],
    "gemm.h::matmul_dab": [
        Arg(name="dy", cxx_type="value::TensorValue"),
        Arg(name="a_or_b", cxx_type="value::TensorValue"),
        Arg(name="transpose_dx", cxx_type="bool", cxx_default=False),
        Arg(name="transpose_dy", cxx_type="bool", cxx_default=False),
    ],
    "nn.h::local_response_norm": [
        Arg(name="x", cxx_type="value::TensorValue"),
        Arg(name="size", cxx_type="int64_t"),
        Arg(name="alpha", cxx_type="double", cxx_default=1e-4),
        Arg(name="beta", cxx_type="double", cxx_default=0.75),
        Arg(name="k", cxx_type="double", cxx_default=1.0),
    ],
    "nn.h::conv_dxw": [
        Arg(name="x_or_w", cxx_type="value::TensorValue"),
        Arg(name="y", cxx_type="value::TensorValue"),
        Arg(name="dy", cxx_type="value::TensorValue"),
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
        Arg(name="x", cxx_type="value::TensorValue"),
        Arg(name="y", cxx_type="value::TensorValue"),
        Arg(name="dy", cxx_type="value::TensorValue"),
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
        Arg(name="x", cxx_type="value::TensorValue"),
        Arg(name="y", cxx_type="value::TensorValue"),
        Arg(name="dy", cxx_type="value::TensorValue"),
        Arg(name="axis", cxx_type="int", cxx_default=-1),
    ],
    "loss.h::loss": [
        Arg(name="y_true", cxx_type="value::TensorValue"),
        Arg(name="y_pred", cxx_type="value::TensorValue"),
    ],
    "loss.h::loss_dx": [
        Arg(name="loss", cxx_type="value::TensorValue"),
        Arg(name="y_true", cxx_type="value::TensorValue"),
        Arg(name="y_pred", cxx_type="value::TensorValue"),
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
        Arg(name="y", cxx_type="value::TensorValue"),
        Arg(name="dy", cxx_type="value::TensorValue"),
    ],
    "ufunc.h::binary_dx": [
        Arg(name="x1", cxx_type="value::Value"),
        Arg(name="x2", cxx_type="value::Value"),
        Arg(name="y", cxx_type="value::TensorValue"),
        Arg(name="dy", cxx_type="value::TensorValue"),
    ],
    "ufunc.h::ternary_dx": [
        Arg(name="x1", cxx_type="value::Value"),
        Arg(name="x2", cxx_type="value::Value"),
        Arg(name="x3", cxx_type="value::Value"),
        Arg(name="y", cxx_type="value::TensorValue"),
        Arg(name="dy", cxx_type="value::TensorValue"),
    ],
    "nn.h::bias_add": [
        Arg(name="x", cxx_type="value::TensorValue"),
        Arg(name="b", cxx_type="value::TensorValue"),
        Arg(name="axis", cxx_type="int", cxx_default=1),
    ],
    "nn.h::bias_add_db": [
        Arg(name="b", cxx_type="value::TensorValue"),
        Arg(name="dy", cxx_type="value::TensorValue"),
        Arg(name="axis", cxx_type="int", cxx_default=1),
    ],
    "likes.h::collapse_like": [
        Arg(name="x", cxx_type="value::TensorValue"),
        Arg(name="shape",
            cxx_type="std::vector<int64_t>",
            cxx_normalizer="IntTuple"),
    ],
    "likes.h::reshape_like": [
        Arg(name="x", cxx_type="value::TensorValue"),
        Arg(name="shape",
            cxx_type="std::vector<int64_t>",
            cxx_normalizer="IntTuple"),
    ],
    "optimizer.h::sgd": [
        Arg(name="x", cxx_type="value::TensorValue"),
        Arg(name="dx", cxx_type="value::TensorValue"),
        Arg(name="v", cxx_type="value::TensorValue"),
        Arg(name="learning_rate", cxx_type="double"),
        Arg(name="mu", cxx_type="double"),
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
