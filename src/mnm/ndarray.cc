#include <mnm/ndarray.h>
#include <mnm/registry.h>

namespace mnm {
namespace ndarray {

using mnm::ndarray::NDArray;
using mnm::registry::Registry;
using mnm::rly::Array;
using mnm::rly::DataType;
using mnm::rly::Expr;
using mnm::rly::IndexExpr;
using mnm::rly::Integer;
using mnm::rly::make_node;
using mnm::rly::NodePtr;

#define MNM_REGISTER_OP(OpName, RegName) \
  MNM_REGISTER_GLOBAL("mnm._ndarray." RegName).set_body_typed(OpName);

#define MNM_DEFINE_UNARY_OP(OpName, RegName, RelayOpMakeName)                      \
  static NDArray OpName(NDArray x) {                                               \
    static const auto* op_make = Registry::Get("relay.op._make." RelayOpMakeName); \
    NodePtr<NDArrayNode> n = make_node<NDArrayNode>();                             \
    n->data = nullptr;                                                             \
    n->expr = (*op_make)(NDArrayNode::Impl::GetExpr(x));                           \
    return NDArray(n);                                                             \
  }                                                                                \
  MNM_REGISTER_OP(OpName, RegName)

#define MNM_DEFINE_BINARY_OP(OpName, RegName, RelayOpMakeName)                            \
  static NDArray OpName(NDArray x1, NDArray x2) {                                         \
    static const auto* op_make = Registry::Get("relay.op._make." RelayOpMakeName);        \
    NodePtr<NDArrayNode> n = make_node<NDArrayNode>();                                    \
    n->data = nullptr;                                                                    \
    n->expr = (*op_make)(NDArrayNode::Impl::GetExpr(x1), NDArrayNode::Impl::GetExpr(x2)); \
    return NDArray(n);                                                                    \
  }                                                                                       \
  MNM_REGISTER_OP(OpName, RegName)

#define MNM_DEFINE_REDUCE_OP(OpName, RegName, RelayOpMakeName)                         \
  static NDArray OpName(NDArray x, Array<Integer> axis, bool keepdims, bool exclude) { \
    static const auto* op_make = Registry::Get("relay.op._make." RelayOpMakeName);     \
    NodePtr<NDArrayNode> n = make_node<NDArrayNode>();                                 \
    n->data = nullptr;                                                                 \
    n->expr = (*op_make)(NDArrayNode::Impl::GetExpr(x), axis, keepdims, exclude);      \
    return NDArray(n);                                                                 \
  }                                                                                    \
  MNM_REGISTER_OP(OpName, RegName)

struct NDArrayNode::Impl final {
  static inline Expr GetExpr(const NDArray& array) {
    return array->expr;
  }

  static NDArray conv2d(NDArray data,                  //
                        NDArray weight,                //
                        Array<IndexExpr> strides,      //
                        Array<IndexExpr> padding,      //
                        Array<IndexExpr> dilation,     //
                        int groups,                    //
                        IndexExpr channels,            //
                        Array<IndexExpr> kernel_size,  //
                        std::string data_layout,       //
                        std::string kernel_layout,     //
                        std::string out_layout,        //
                        DataType out_dtype) {
    static const auto* op_make = Registry::Get("relay.op.nn._make.conv2d");
    NodePtr<NDArrayNode> n = make_node<NDArrayNode>();
    n->data = nullptr;
    n->expr = (*op_make)(data->expr,     //
                         weight->expr,   //
                         strides,        //
                         padding,        //
                         dilation,       //
                         groups,         //
                         channels,       //
                         kernel_size,    //
                         data_layout,    //
                         kernel_layout,  //
                         out_layout,     //
                         out_dtype);
    return NDArray(n);
  }

  static NDArray conv2d_transpose(NDArray data,                     //
                                  NDArray weight,                   //
                                  Array<IndexExpr> strides,         //
                                  Array<IndexExpr> padding,         //
                                  Array<IndexExpr> dilation,        //
                                  int groups,                       //
                                  IndexExpr channels,               //
                                  Array<IndexExpr> kernel_size,     //
                                  std::string data_layout,          //
                                  std::string kernel_layout,        //
                                  Array<IndexExpr> output_padding,  //
                                  DataType out_dtype) {
    static const auto* op_make = Registry::Get("relay.op.nn._make.conv2d_transpose");
    NodePtr<NDArrayNode> n = make_node<NDArrayNode>();
    n->data = nullptr;
    n->expr = (*op_make)(data->expr,      //
                         weight->expr,    //
                         strides,         //
                         padding,         //
                         dilation,        //
                         groups,          //
                         channels,        //
                         kernel_size,     //
                         data_layout,     //
                         kernel_layout,   //
                         output_padding,  //
                         out_dtype);
    return NDArray(n);
  }

  static NDArray deformable_conv2d(NDArray data,                  //
                                   NDArray offset,                //
                                   NDArray weight,                //
                                   Array<IndexExpr> strides,      //
                                   Array<IndexExpr> padding,      //
                                   Array<IndexExpr> dilation,     //
                                   int deformable_groups,         //
                                   int groups,                    //
                                   int channels,                  //
                                   Array<IndexExpr> kernel_size,  //
                                   std::string data_layout,       //
                                   std::string kernel_layout,     //
                                   std::string out_layout,        //
                                   DataType out_dtype) {
    static const auto* op_make = Registry::Get("relay.op.nn._make.deformable_conv2d");
    NodePtr<NDArrayNode> n = make_node<NDArrayNode>();
    n->data = nullptr;
    n->expr = (*op_make)(data->expr,         //
                         offset->expr,       //
                         weight->expr,       //
                         strides,            //
                         padding,            //
                         dilation,           //
                         deformable_groups,  //
                         groups,             //
                         channels,           //
                         kernel_size,        //
                         data_layout,        //
                         kernel_layout,      //
                         out_layout,         //
                         out_dtype);
    return NDArray(n);
  }

  static NDArray bias_add(NDArray data, NDArray bias, int axis) {
    static const auto* op_make = Registry::Get("relay.op.nn._make.bias_add");
    NodePtr<NDArrayNode> n = make_node<NDArrayNode>();
    n->data = nullptr;
    n->expr = (*op_make)(data->expr, bias->expr, axis);
    return NDArray(n);
  }

  static NDArray dense(NDArray data, NDArray weight, IndexExpr units) {
    static const auto* op_make = Registry::Get("relay.op.nn._make.dense");
    NodePtr<NDArrayNode> n = make_node<NDArrayNode>();
    n->data = nullptr;
    n->expr = (*op_make)(data->expr, weight->expr, units);
    return NDArray(n);
  }

  static NDArray leaky_relu(NDArray data, double alpha) {
    static const auto* op_make = Registry::Get("relay.op.nn._make.leaky_relu");
    NodePtr<NDArrayNode> n = make_node<NDArrayNode>();
    n->data = nullptr;
    n->expr = (*op_make)(data->expr, alpha);
    return NDArray(n);
  }

  static NDArray prelu(NDArray data, NDArray alpha, int axis) {
    static const auto* op_make = Registry::Get("relay.op.nn._make.prelu");
    NodePtr<NDArrayNode> n = make_node<NDArrayNode>();
    n->data = nullptr;
    n->expr = (*op_make)(data->expr, alpha->expr, axis);
    return NDArray(n);
  }

  static NDArray softmax(NDArray data, int axis) {
    static const auto* op_make = Registry::Get("relay.op.nn._make.softmax");
    NodePtr<NDArrayNode> n = make_node<NDArrayNode>();
    n->data = nullptr;
    n->expr = (*op_make)(data->expr, axis);
    return NDArray(n);
  }

  static NDArray log_softmax(NDArray data, int axis) {
    static const auto* op_make = Registry::Get("relay.op.nn._make.log_softmax");
    NodePtr<NDArrayNode> n = make_node<NDArrayNode>();
    n->data = nullptr;
    n->expr = (*op_make)(data->expr, axis);
    return NDArray(n);
  }

  static NDArray batch_flatten(NDArray data) {
    static const auto* op_make = Registry::Get("relay.op.nn._make.batch_flatten");
    NodePtr<NDArrayNode> n = make_node<NDArrayNode>();
    n->data = nullptr;
    n->expr = (*op_make)(data->expr);
    return NDArray(n);
  }

  static NDArray relu(NDArray data) {
    static const auto* op_make = Registry::Get("relay.op.nn._make.relu");
    NodePtr<NDArrayNode> n = make_node<NDArrayNode>();
    n->data = nullptr;
    n->expr = (*op_make)(data->expr);
    return NDArray(n);
  }

  static NDArray lrn(NDArray data, int size, int axis, double alpha, double beta, double bias) {
    static const auto* op_make = Registry::Get("relay.op.nn._make.lrn");
    NodePtr<NDArrayNode> n = make_node<NDArrayNode>();
    n->data = nullptr;
    n->expr = (*op_make)(data->expr, size, axis, alpha, beta, bias);
    return NDArray(n);
  }

  static NDArray l2_normalize(NDArray data, double eps, Array<Integer> axis) {
    static const auto* op_make = Registry::Get("relay.op.nn._make.l2_normalize");
    NodePtr<NDArrayNode> n = make_node<NDArrayNode>();
    n->data = nullptr;
    n->expr = (*op_make)(data->expr, eps, axis);
    return NDArray(n);
  }

  static NDArray batch_norm(NDArray data,         //
                            NDArray gamma,        //
                            NDArray beta,         //
                            NDArray moving_mean,  //
                            NDArray moving_var,   //
                            int axis,             //
                            double epsilon,       //
                            bool center,          //
                            bool scale) {
    static const auto* op_make = Registry::Get("relay.op.nn._make.batch_norm");
    NodePtr<NDArrayNode> n = make_node<NDArrayNode>();
    n->data = nullptr;
    n->expr = (*op_make)(data->expr,         //
                         gamma->expr,        //
                         beta->expr,         //
                         moving_mean->expr,  //
                         moving_var->expr,   //
                         axis,               //
                         epsilon,            //
                         center,             //
                         scale);
    return NDArray(n);
  }

  static NDArray batch_matmul(NDArray x1, NDArray x2) {
    static const auto* op_make = Registry::Get("relay.op.nn._make.batch_matmul");
    NodePtr<NDArrayNode> n = make_node<NDArrayNode>();
    n->data = nullptr;
    n->expr = (*op_make)(x1->expr, x2->expr);
    return NDArray(n);
  }

  static NDArray max_pool2d(NDArray data,                //
                            Array<IndexExpr> pool_size,  //
                            Array<IndexExpr> strides,    //
                            std::string layout,          //
                            Array<IndexExpr> padding,    //
                            bool ceil_mode) {
    static const auto* op_make = Registry::Get("relay.op.nn._make.max_pool2d");
    NodePtr<NDArrayNode> n = make_node<NDArrayNode>();
    n->data = nullptr;
    n->expr = (*op_make)(data->expr, pool_size, strides, layout, padding, ceil_mode);
    return NDArray(n);
  }

  static NDArray avg_pool2d(NDArray data,                //
                            Array<IndexExpr> pool_size,  //
                            Array<IndexExpr> strides,    //
                            Array<IndexExpr> padding,    //
                            std::string layout,          //
                            bool ceil_mode,              //
                            bool count_include_pad) {
    static const auto* op_make = Registry::Get("relay.op.nn._make.avg_pool2d");
    NodePtr<NDArrayNode> n = make_node<NDArrayNode>();
    n->data = nullptr;
    n->expr = (*op_make)(data->expr,  //
                         pool_size,   //
                         strides,     //
                         padding,     //
                         layout,      //
                         ceil_mode,   //
                         count_include_pad);
    return NDArray(n);
  }

  static NDArray cast(NDArray data, DataType dtype) {
    static const auto* op_make = Registry::Get("relay._make.cast");
    NodePtr<NDArrayNode> n = make_node<NDArrayNode>();
    n->data = nullptr;
    n->expr = (*op_make)(data->expr, dtype);
    return NDArray(n);
  }

  static NDArray expand_dims(NDArray data, int axis, int num_newaxis) {
    static const auto* op_make = Registry::Get("relay.op._make.expand_dims");
    NodePtr<NDArrayNode> n = make_node<NDArrayNode>();
    n->data = nullptr;
    n->expr = (*op_make)(data->expr, axis, num_newaxis);
    return NDArray(n);
  }

  static NDArray concatenate(NDArray data, int axis) {
    static const auto* op_make = Registry::Get("relay.op._make.concatenate");
    NodePtr<NDArrayNode> n = make_node<NDArrayNode>();
    n->data = nullptr;
    n->expr = (*op_make)(data->expr, axis);
    return NDArray(n);
  }

  static NDArray transpose(NDArray data, Array<Integer> axes) {
    static const auto* op_make = Registry::Get("relay.op._make.transpose");
    NodePtr<NDArrayNode> n = make_node<NDArrayNode>();
    n->data = nullptr;
    n->expr = (*op_make)(data->expr, axes);
    return NDArray(n);
  }

  static NDArray reshape(NDArray data, Array<Integer> newshape) {
    static const auto* op_make = Registry::Get("relay.op._make.reshape");
    NodePtr<NDArrayNode> n = make_node<NDArrayNode>();
    n->data = nullptr;
    n->expr = (*op_make)(data->expr, newshape);
    return NDArray(n);
  }

  static NDArray reshape_like(NDArray data, NDArray shape_like) {
    static const auto* op_make = Registry::Get("relay.op._make.reshape_like");
    NodePtr<NDArrayNode> n = make_node<NDArrayNode>();
    n->data = nullptr;
    n->expr = (*op_make)(data->expr, shape_like->expr);
    return NDArray(n);
  }

  static NDArray take(NDArray data, NDArray indices, Integer axis, std::string mode) {
    static const auto* op_make = Registry::Get("relay.op._make.take");
    NodePtr<NDArrayNode> n = make_node<NDArrayNode>();
    n->data = nullptr;
    n->expr = (*op_make)(data->expr, indices->expr, axis, mode);
    return NDArray(n);
  }

  static NDArray full(NDArray fill_value, Array<IndexExpr> shape, DataType dtype) {
    static const auto* op_make = Registry::Get("relay.op._make.full");
    NodePtr<NDArrayNode> n = make_node<NDArrayNode>();
    n->data = nullptr;
    n->expr = (*op_make)(fill_value->expr, shape, dtype);
    return NDArray(n);
  }

  static NDArray zeros(Array<IndexExpr> shape, DataType dtype) {
    static const auto* op_make = Registry::Get("relay.op._make.zeros");
    NodePtr<NDArrayNode> n = make_node<NDArrayNode>();
    n->data = nullptr;
    n->expr = (*op_make)(shape, dtype);
    return NDArray(n);
  }

  static NDArray ones(Array<IndexExpr> shape, DataType dtype) {
    static const auto* op_make = Registry::Get("relay.op._make.ones");
    NodePtr<NDArrayNode> n = make_node<NDArrayNode>();
    n->data = nullptr;
    n->expr = (*op_make)(shape, dtype);
    return NDArray(n);
  }

  static NDArray full_like(NDArray data, NDArray fill_value) {
    static const auto* op_make = Registry::Get("relay.op._make.full_like");
    NodePtr<NDArrayNode> n = make_node<NDArrayNode>();
    n->data = nullptr;
    n->expr = (*op_make)(data->expr, fill_value->expr);
    return NDArray(n);
  }

  static NDArray arange(NDArray start, NDArray stop, NDArray step, DataType dtype) {
    static const auto* op_make = Registry::Get("relay.op._make.arange");
    NodePtr<NDArrayNode> n = make_node<NDArrayNode>();
    n->data = nullptr;
    n->expr = (*op_make)(start->expr, stop->expr, step->expr, dtype);
    return NDArray(n);
  }

  static NDArray repeat(NDArray data, int repeats, int axis) {
    static const auto* op_make = Registry::Get("relay.op._make.repeat");
    NodePtr<NDArrayNode> n = make_node<NDArrayNode>();
    n->data = nullptr;
    n->expr = (*op_make)(data->expr, repeats, axis);
    return NDArray(n);
  }

  static NDArray tile(NDArray data, Array<Integer> reps) {
    static const auto* op_make = Registry::Get("relay.op._make.tile");
    NodePtr<NDArrayNode> n = make_node<NDArrayNode>();
    n->data = nullptr;
    n->expr = (*op_make)(data->expr, reps);
    return NDArray(n);
  }

  static NDArray reverse(NDArray data, int axis) {
    static const auto* op_make = Registry::Get("relay.op._make.reverse");
    NodePtr<NDArrayNode> n = make_node<NDArrayNode>();
    n->data = nullptr;
    n->expr = (*op_make)(data->expr, axis);
    return NDArray(n);
  }

  static NDArray where(NDArray condition, NDArray x, NDArray y) {
    static const auto* op_make = Registry::Get("relay.op._make.where");
    NodePtr<NDArrayNode> n = make_node<NDArrayNode>();
    n->data = nullptr;
    n->expr = (*op_make)(condition->expr, x->expr, y->expr);
    return NDArray(n);
  }

  static NDArray squeeze(NDArray data, Array<Integer> axis) {
    static const auto* op_make = Registry::Get("relay.op._make.squeeze");
    NodePtr<NDArrayNode> n = make_node<NDArrayNode>();
    n->data = nullptr;
    n->expr = (*op_make)(data->expr, axis);
    return NDArray(n);
  }

  static NDArray broadcast_to(NDArray data, Array<IndexExpr> shape) {
    static const auto* op_make = Registry::Get("relay.op._make.broadcast_to");
    NodePtr<NDArrayNode> n = make_node<NDArrayNode>();
    n->data = nullptr;
    n->expr = (*op_make)(data->expr, shape);
    return NDArray(n);
  }

  static NDArray gather_nd(NDArray data, NDArray indices) {
    static const auto* op_make = Registry::Get("relay.op._make.gather_nd");
    NodePtr<NDArrayNode> n = make_node<NDArrayNode>();
    n->data = nullptr;
    n->expr = (*op_make)(data->expr, indices->expr);
    return NDArray(n);
  }

};  // namespace ndarray

MNM_DEFINE_UNARY_OP(log, "log", "log");
MNM_DEFINE_UNARY_OP(exp, "exp", "exp");
MNM_DEFINE_UNARY_OP(sqrt, "sqrt", "sqrt");
MNM_DEFINE_UNARY_OP(zeros_like, "zeros_like", "zeros_like");
MNM_DEFINE_UNARY_OP(ones_like, "ones_like", "ones_like");
MNM_DEFINE_UNARY_OP(sigmoid, "sigmoid", "sigmoid");
MNM_DEFINE_UNARY_OP(copy, "copy", "copy");
MNM_DEFINE_UNARY_OP(floor, "floor", "floor");
MNM_DEFINE_UNARY_OP(ceil, "ceil", "ceil");
MNM_DEFINE_UNARY_OP(trunc, "trunc", "trunc");
MNM_DEFINE_UNARY_OP(round, "round", "round");
MNM_DEFINE_UNARY_OP(sign, "sign", "sign");
MNM_DEFINE_UNARY_OP(abs, "abs", "abs");
MNM_DEFINE_UNARY_OP(tanh, "tanh", "tanh");
MNM_DEFINE_UNARY_OP(negative, "negative", "negative");
MNM_DEFINE_UNARY_OP(logical_not, "logical_not", "logical_not");
MNM_DEFINE_BINARY_OP(add, "add", "add");
MNM_DEFINE_BINARY_OP(subtract, "subtract", "subtract");
MNM_DEFINE_BINARY_OP(right_shift, "right_shift", "right_shift");
MNM_DEFINE_BINARY_OP(left_shift, "left_shift", "left_shift");
MNM_DEFINE_BINARY_OP(maximum, "maximum", "maximum");
MNM_DEFINE_BINARY_OP(minimum, "minimum", "minimum");
MNM_DEFINE_BINARY_OP(divide, "divide", "divide");
MNM_DEFINE_BINARY_OP(multiply, "multiply", "multiply");
MNM_DEFINE_BINARY_OP(power, "power", "power");
MNM_DEFINE_BINARY_OP(mod, "mod", "mod");
MNM_DEFINE_BINARY_OP(logical_and, "logical_and", "logical_and");
MNM_DEFINE_BINARY_OP(logical_or, "logical_or", "logical_or");
MNM_DEFINE_BINARY_OP(equal, "equal", "equal");
MNM_DEFINE_BINARY_OP(not_equal, "not_equal", "not_equal");
MNM_DEFINE_BINARY_OP(less, "less", "less");
MNM_DEFINE_BINARY_OP(less_equal, "less_equal", "less_equal");
MNM_DEFINE_BINARY_OP(greater, "greater", "greater");
MNM_DEFINE_BINARY_OP(greater_equal, "greater_equal", "greater_equal");
MNM_DEFINE_REDUCE_OP(argmax, "argmax", "argmax");
MNM_DEFINE_REDUCE_OP(argmin, "argmin", "argmin");
MNM_DEFINE_REDUCE_OP(sum, "sum", "sum");
MNM_DEFINE_REDUCE_OP(max, "max", "max");
MNM_DEFINE_REDUCE_OP(min, "min", "min");
MNM_DEFINE_REDUCE_OP(prod, "prod", "prod");
MNM_DEFINE_REDUCE_OP(mean, "mean", "mean");
MNM_REGISTER_OP(NDArrayNode::Impl::conv2d, "conv2d");
MNM_REGISTER_OP(NDArrayNode::Impl::conv2d_transpose, "conv2d_transpose");
MNM_REGISTER_OP(NDArrayNode::Impl::deformable_conv2d, "deformable_conv2d");
MNM_REGISTER_OP(NDArrayNode::Impl::bias_add, "bias_add");
MNM_REGISTER_OP(NDArrayNode::Impl::dense, "dense");
MNM_REGISTER_OP(NDArrayNode::Impl::leaky_relu, "leaky_relu");
MNM_REGISTER_OP(NDArrayNode::Impl::prelu, "prelu");
MNM_REGISTER_OP(NDArrayNode::Impl::softmax, "softmax");
MNM_REGISTER_OP(NDArrayNode::Impl::log_softmax, "log_softmax");
MNM_REGISTER_OP(NDArrayNode::Impl::batch_flatten, "batch_flatten");
MNM_REGISTER_OP(NDArrayNode::Impl::relu, "relu");
MNM_REGISTER_OP(NDArrayNode::Impl::lrn, "lrn");
MNM_REGISTER_OP(NDArrayNode::Impl::l2_normalize, "l2_normalize");
MNM_REGISTER_OP(NDArrayNode::Impl::batch_norm, "batch_norm");
MNM_REGISTER_OP(NDArrayNode::Impl::batch_matmul, "batch_matmul");
MNM_REGISTER_OP(NDArrayNode::Impl::max_pool2d, "max_pool2d");
MNM_REGISTER_OP(NDArrayNode::Impl::avg_pool2d, "avg_pool2d");
MNM_REGISTER_OP(NDArrayNode::Impl::cast, "cast");
MNM_REGISTER_OP(NDArrayNode::Impl::expand_dims, "expand_dims");
MNM_REGISTER_OP(NDArrayNode::Impl::concatenate, "concatenate");
MNM_REGISTER_OP(NDArrayNode::Impl::transpose, "transpose");
MNM_REGISTER_OP(NDArrayNode::Impl::reshape, "reshape");
MNM_REGISTER_OP(NDArrayNode::Impl::reshape_like, "reshape_like");
MNM_REGISTER_OP(NDArrayNode::Impl::take, "take");
MNM_REGISTER_OP(NDArrayNode::Impl::full, "full");
MNM_REGISTER_OP(NDArrayNode::Impl::zeros, "zeros");
MNM_REGISTER_OP(NDArrayNode::Impl::ones, "ones");
MNM_REGISTER_OP(NDArrayNode::Impl::full_like, "full_like");
MNM_REGISTER_OP(NDArrayNode::Impl::arange, "arange");
MNM_REGISTER_OP(NDArrayNode::Impl::repeat, "repeat");
MNM_REGISTER_OP(NDArrayNode::Impl::tile, "tile");
MNM_REGISTER_OP(NDArrayNode::Impl::reverse, "reverse");
MNM_REGISTER_OP(NDArrayNode::Impl::where, "where");
MNM_REGISTER_OP(NDArrayNode::Impl::squeeze, "squeeze");
MNM_REGISTER_OP(NDArrayNode::Impl::broadcast_to, "broadcast_to");
MNM_REGISTER_OP(NDArrayNode::Impl::gather_nd, "gather_nd");

}  // namespace ndarray
}  // namespace mnm
