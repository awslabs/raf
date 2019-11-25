import def_schema
from codegen_utils import Op

OPS = [
    Op(name="conv2d", schema_name="conv"),
    Op(name="max_pool2d", schema_name="pool"),
    Op(name="avg_pool2d", schema_name="pool"),
    Op(name="softmax", schema_name="softmax"),
    Op(name="log_softmax", schema_name="softmax"),
    Op(name="batch_norm_train", schema_name="batch_norm"),
    Op(name="batch_norm_infer", schema_name="batch_norm"),
    Op(name="conv2d_dx", schema_name="conv_dxw"),
    Op(name="conv2d_dw", schema_name="conv_dxw"),
    Op(name="max_pool2d_dx", schema_name="pool_dx"),
    Op(name="avg_pool2d_dx", schema_name="pool_dx"),
    Op(name="softmax_dx", schema_name="softmax_dx"),
    Op(name="log_softmax_dx", schema_name="softmax_dx"),
    Op(name="batch_flatten", schema_name="unary"),
    Op(name="negative", schema_name="unary_ufunc"),
    Op(name="logical_not", schema_name="unary_ufunc"),
    Op(name="relu", schema_name="unary"),
    Op(name="tanh", schema_name="unary"),
    Op(name="sigmoid", schema_name="unary"),
    Op(name="relu_dx", schema_name="unary_dx"),
    Op(name="tanh_dx", schema_name="unary_dx"),
    Op(name="sigmoid_dx", schema_name="unary_dx"),
    Op(name="add", schema_name="binary_ufunc"),
    Op(name="subtract", schema_name="binary_ufunc"),
    Op(name="multiply", schema_name="binary_ufunc"),
    Op(name="divide", schema_name="binary_ufunc"),
    Op(name="mod", schema_name="binary_ufunc"),
    Op(name="less", schema_name="binary_ufunc"),
    Op(name="greater", schema_name="binary_ufunc"),
    Op(name="less_equal", schema_name="binary_ufunc"),
    Op(name="greater_equal", schema_name="binary_ufunc"),
    Op(name="equal", schema_name="binary_ufunc"),
    Op(name="not_equal", schema_name="binary_ufunc"),
    Op(name="linear", schema_name="binary"),
]


def by_name():
    result = dict()
    schemas = def_schema.by_name()

    for op in OPS:
        op.schema = schemas[op.schema_name]
        result[op.name] = op

    return result
