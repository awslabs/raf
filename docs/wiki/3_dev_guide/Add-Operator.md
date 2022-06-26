<!--- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Add an Operator

This article introduces the process of adding a new operator to RAF. It includes 3 major steps:
1. Define the operator.
2. Implement the operator.
3. Write unit tests. 

We will use `softmax` as a running example in this article to help you better understand the process.

## 1. Define an Operator

### 1.a. Define Schema

Before defining a new operator, you need to first have an operator schema, which includes the operator arguments and their types. You can check [scripts/src_codegen/def_schema.py](https://github.com/awslabs/raf/blob/main/scripts/src_codegen/def_schema.py) to see if any of existing schema fits your operator, and simply reuse it if so. In this case, you can proceed to the next section directly to define the operator with the schema.

If you are not lucky enough to reuse an existing schmea, you have to define a new schema for this operator. If the operator is available in TVM/Relay, you can and are encouraged to refer to TVM's implementation. You can search `RELAY_REGISTER_OP(relay_op_name)` in the TVM code base to check its availability.

For example, here is the softmax in Relay [link](https://github.com/apache/tvm/blob/15bdf28209e5f2bcb5ffc21bd56b3ae428cf2da7/src/relay/op/nn/nn.cc#L363). We can see that the softmax operator takes one argument (`data`) and an attribute (`SoftmaxAttrs`). Attributes are fixed arguments which are supposted to be known at compile time (refer to [Adding an Operator to Relay](https://tvm.apache.org/docs/dev/how_to/relay_add_op.html) for more details).
Note that operators in RAF does not have attributes but only arguments. As a result, we define the schema of softmax in [scripts/src_codegen/def_schema.py](https://github.com/awslabs/raf/blob/64d10b6118ca23f04cc9441a4e531d549eb575b3/scripts/src_codegen/def_schema.py#L120) as follows:

```python
"nn.h::softmax": [
    Arg(name="x", cxx_type="value::BaseTensorValue"),
    Arg(name="axis", cxx_type="int", cxx_default=-1),
],
```

As can be seen, we do not directly write C++ code to define a schema. Instead, we write the RAF-specific DSL, and then run the code generator to generate the C++ and Python code:

```
scripts/src_codegen/run_all.sh
```

In the above schema, `nn.h` is the place for generated C++ implementation, so we can find the generated code in `src/op/schema/nn.h`:

```c++
class SoftmaxArgs : public ir::AttrsNode<SoftmaxArgs> {
 public:
  value::BaseTensorValue x;
  int axis{-1};
  RAF_OP_SCHEMA(SoftmaxArgs, "raf.args.softmax");
};
```

Note that `src/op/schema/nn.h` is auto-generated. If you search in the Github repository, you won't find it.

You can also find the schema registration in `src/op/regs/regs.cc`. These are the implementations about how this schema should be processed in the IR graph.

```c++
Attrs Softmax(const TVMArgs& values, GradTape* tapes) {
  RAF_PRELUDE(schema::SoftmaxArgs, 2);  // NOLINT(whitespace/line_length)
  RAF_TAPE(0, ffi2schema::Tensor, x);
  RAF_POD(1, ffi2schema::Int, axis);
  return Attrs(attrs);
}

Array<Expr> Softmax(const TVMArgs& values) {
  RAF_PRELUDE(2);
  RAF_ARG(0, ffi2expr::Tensor, x);
  RAF_ARG(1, ffi2expr::Int, axis);
  RAF_RET();
}

template <const char* op_name>
Attrs Softmax(const Array<Value>& values) {
  RAF_PRELUDE(1, 2, schema::SoftmaxArgs);
  RAF_REQUIRED(0, value2schema::Tensor, x);
  RAF_OPTIONAL(1, value2schema::Int, axis);
  return Attrs(attrs);
}

RAF_REGISTER_OBJECT_REFLECT(SoftmaxArgs);
```

### 1.b. Define Operator Symbol

Now let's define the `softmax` operator. In this example, we define a `softmax` operator and bind it to the schema we just defined. We just need to add one line to [scripts/src_codegen/def_op.py](https://github.com/awslabs/raf/blob/main/scripts/src_codegen/def_op.py):

```python
Op(name="softmax", schema_name="softmax"),
```

Be aware that a schema is just a list of input arguments and their types instead of an operator. It means that one schema can be used by many operators as long as their input arguments are the same. For example, as you can see, the `softmax` schema is also used by a similar operator, `log_softmax`:

```python
Op(name="log_softmax", schema_name="softmax"),
```

Let's run [scripts/src_codegen/run_all.sh](https://github.com/awslabs/raf/blob/main/scripts/src_codegen/run_all.sh) again to see what will be generated in `src/op/regs/regs.cc`:

```c++
static const char softmax[] = "raf.op.softmax";

RAF_REGISTER_GLOBAL("raf.op.imp.softmax").set_body([](TVMArgs args, TVMRetValue* ret) {
  RAF_PRELUDE(softmax, 2, ffi2schema::Softmax,
              schema::SoftmaxArgs);  // NOLINT(whitespace/line_length)
  RAF_SET_ENV(vpack->x[0], schema2value::Tensor(schema->x));
  RAF_SET_ENV(vpack->x[1], schema2value::Int(schema->axis));
  RAF_SET_ENV(vpack->y, value);
  *ret = RAF_RET();
});

RAF_REGISTER_GLOBAL("raf.op.sym.softmax").set_body(RAF_SYMBOLIC_API(softmax, 2, Softmax));

RAF_BIND_SCHEMA("raf.op.softmax", names::softmax,
                value2schema::Softmax);  // NOLINT(whitespace/line_length)
RAF_BIND_SCHEMA_FIELD_INDEX("raf.op.softmax", names::softmax,
                            schema_field_idx::Softmax);  // NOLINT(whitespace/line_length)
```

It registers an operator named `softmax` and binds this operator to the schema we defined. It also creates the FFI and Python bindings for this operator.

### Checkpoint: Build and Run the Operator

Let’s now check if `softmax` has been successfully added to the system and can be accessed via Python API:

```bash
$ cd build; cmake ..; make -j$(nproc)
$ python3
$ >>> import raf
$ >>> raf.softmax
<function softmax at 0x7fd2c7af7158>
```

It works! Unfortunately, if we try to invoke the operator, it will raise an error complaining that `FRAFDecalre` does not exist:

```bash
>>> import numpy as np
>>> x = raf.array(np.random.randn(1, 2).astype("float32"), device="cpu")
>>> raf.softmax(x, axis=0)
Attribute FRAFDeclare has not been registered for Operator raf.op.softmax
```

This is because we have not defined the behavior of this operator, so RAF does not know how to invoke it. 
We will define it in the next step.

### 1.c. Declare the Behavior

Now let’s declare the behavior of the operator. There are two major behaviors to define:

1. The shape of the output (inferred from the inputs’ shapes).
2. The output value, or the method to produce the value,


If this operator is compute intensive (e.g., `softmax`) and has to be offloaded to a
backend (CUDA, CuBLAS, CuDNN, TVM, LLVM, etc.), we only need to assign a tensor placeholder (with inferred shape, data type and device context) to `call->out` to let RAF know how its output should be. Here is how we declare `softmax` in [src/op/declare/nn.cc](https://github.com/awslabs/raf/blob/main/src/op/declare/nn.cc):

```c++
void Softmax(const CallValues& call) {
  const auto* args = call->args.as<SoftmaxArgs>();
  CHECK(args != nullptr);
  const DLTensor* x = args->x;
  std::vector<int64_t> shape(x->shape, x->shape + x->ndim);
  NormalizeAxis(args->axis, x->ndim);
  call->out = TensorValue::Assemble(/*dev=*/x->device,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/shape);
  call->device = x->device;
}
RAF_OP_DECLARE("raf.op.softmax", Softmax);
```

On the other hand, if the result can be simply computed and does not need to be offloaded to a backend, then we can directly assign the results to `call->out` in `RAF_OP_DECLARE` and set `call->callee = ir::NullValue<OpValue>();` to let RAF know that this operator do not have a callee and `call->out` is already the valid output. For example, `shape`, `get_reduce_axis`, and scalar versions of all kinds of unary/binary operators fall into this category. Here is an example of `shape`'s declaration:

```c++
void Shape(const CallValues& call) {
  const auto* args = call->args.as<UnaryArgs>();
  CHECK(args != nullptr);
  auto x_type = op::type::GetType(args->x);
  std::vector<Value> shape;
  if (auto* t_type = x_type.as<ir::TensorTypeNode>()) {
      for (auto ty : t_type->shape) {
      shape.push_back(ScalarValue::make(ty.as<ir::IntImmNode>()->value));
      }
      call->out = TupleValue::make(shape);
  } else {
      call->out = ir::NullValue<Value>();
  }
  call->callee = ir::NullValue<OpValue>();
}
RAF_OP_DECLARE("raf.op.shape", Shape);
```

### 1.d. Define Type Function

In addition to the declaration, we also need to define a type function of an operator. Type function is used by RAF optimization passes to infer the types of each tensor in an IR graph. For example, the second argument of `softmax` (i.e., `axis`) has to be in `int` type. Here is how we implement the type function of `softmax` in [src/op/ty/nn.cc](https://github.com/awslabs/raf/blob/main/src/op/ty/nn.cc):

```c++
template <typename T>
Type GeneralAxisInfer(const CallValues& value) {
  const auto* args = value->args.as<T>();
  CHECK(args != nullptr);
  TensorType x = Downcast<TensorType>(GetType(args->x));
  int axis = args->axis;
  int ndim = x->shape.size();
  CHECK(-ndim <= axis && axis < ndim)
      << "ValueError: invalid axis = " << axis << " on ndim = " << ndim;
  return x;
}

RAF_OP_TYPE("raf.op.softmax", "Softmax", GeneralAxisInfer<SoftmaxArgs>);
```

The type function accepts a `value`, which includes the real arguments of this operator in an IR graph. Then it 
1. validates the input arguments
2. returns the inferred output type `x`. 

Since the output type of `softmax` is the same as its input data, we simply return the type of the input.

### 1.e. Define Gradient Function

Since RAF supports model training and auto differentiation, we need to define a gradient function for each forward operator to let RAF generate the logic for gradient computation. Here is the gradient function of `softmax` in [src/op/grad/nn.cc](https://github.com/awslabs/raf/blob/main/src/op/grad/nn.cc):

```c++
template <const char* GradOp>
Array<Expr> SoftmaxGradImpl(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                            const Expr& dy) {
  static auto op_dx = Op::Get(GradOp);
  const CallNode* call = orig_call.as<CallNode>();
  const Expr& axis = call->args[1];
  return {Call(op_dx, {y, dy, axis})};
}

RAF_OP_GRAD("raf.op.softmax", SoftmaxGradImpl);
```

The gradient function takes in the forward call node `orig_call` and its argument `orig_args`, forward output `y` and backward input `dy`. The function then generates an expression of gradient calculation.

Please note that although the gradient is usually calculated with `x` (from `orig_args`), `y` and `dy` intuitively, the gradient of some operators can be obtained without `x` or `y`. You should evaluate carefully whether you really need both `x` and `y` to compute the gradient, because in this case, we have to keep both tensors from forward to backward, resulting in a larger memory footprint. For the softmax example, we only need `y` to compute the gradient. Therefore, the return statement of the gradient function is:

```c++
return {Call(op_dx, {y, dy, axis})};
```

In this case, we can free the tensor `x` right after the forward op.

Now let's introduce the two approaches for gradient expression implementations.

#### a. Write a compute expression

You can write a compute expression using [(tensor expression) TE](https://tvm.apache.org/docs/tutorial/tensor_expr_get_started.html) (a TVM DSL to describe a computation graph) as you may have done for the forward operator. In this way, the backward logic in an IR graph would be a series of small operators. We usually choose this approach for two reasons:

1. The gradient compute is simple, such as `add` in [src/op/grad/binary.cc](https://github.com/awslabs/raf/blob/dab4d40a892619aaf85bc0211825139cb03d017f/src/op/grad/binary.cc#L20).
2. The gradient compute can be decomposed to provide more fusion opportunities, such as `rsqrt` in [src/op/grad/nn.cc](https://github.com/awslabs/raf/blob/dab4d40a892619aaf85bc0211825139cb03d017f/src/op/grad/nn.cc#L180).

#### b. Define a backward operator

Alternatively, you can also define another **backward** operator and use it here. In this way, the backward logic in an IR graph would be a single operator. We usually choose this approach when the backward logic has special optimizations. For example:

1. The backward logic is supported by kernel libraries, such as `conv2d_dx`.
2. By re-computing some operators (usually `add` or `multiply`) in the backward logic instead of using the forward tensors, we can reduce the memory footprint.

Taking `softmax` as an example, we define and register its gradient in [src/op/grad/nn.cc](https://github.com/awslabs/raf/blob/main/src/op/grad/nn.cc):

```c++
Array<Expr> SoftmaxGradImpl(const Expr& orig_call, const Array<Expr> orig_args, const Var& y,
                            const Expr& dy) {
  static auto op_dx = Op::Get("raf.op.softmax_dx");
  const CallNode* call = orig_call.as<CallNode>();
  const Expr& x = call->args[0];
  const Expr& axis = call->args[1];
  return {Call(op_dx, {y, dy, axis})};
}

RAF_OP_GRAD("raf.op.softmax", SoftmaxGradImpl);
```

Since `softmax_dx` can be offloaded to CuDNN or optimized by TVM, we define this backward operator and use it here. Its schema is `softmax_dx(y, dy, axis)`.


## 2. Implement Dialect Operators

We now have a well-defined `softmax`. Let’s build RAF and run the operator:

```bash
>>> raf.softmax(x, axis=0)
ValueError: Cannot dispatch raf.op.softmax@cpu(0)
```

Oops, we still get an error...Recall that we attempt to offload this operator to a backend when implementing its declaration, so we put a placeholder to its `call->out`. When RAF sees a placeholder in `call->out`, it tries to dispatch this operator to one of the available backends as the callee for execution. However, since we have not defined any backend for this operator, the dispatching failed.

Formally, the operator we just defined is named **base operator** in RAF, which includes backend-independent schema and attributes. Meanwhile, the backend-specific operators are named **dialect operators**. One base operator can be associated with multiple dialect operators, and each of them is in charge of one backend execution. For example, the base operator `raf.op.softmax` has the following dialect operators in RAF:

```
raf.op.softmax
 |- raf.op.tvm.softmax
 |- raf.op.cudnn.softmax
```

where `raf.op.tvm.softmax` dispatchs the operator to TVM to generate LLVM or CUDA code for CPUs and GPUs; `raf.op.cudnn.softmax` dispatchs the operator to CuDNN for GPUs. In this section, we introduce how to implement dialect operators in RAF.

### 2.a. Relay/TVM Dialect

Every base operator should have a Relay/TVM dialect operator implementation as the default option for dispatching, because TVM can generate the binary executable for any hardware platform, given a compute and schedule definition.

#### Case 1: The operator has an implementation in Relay and has a Relay op strategy

If an operator has a corresponding implementation in Relay, then we can simply add one line to [scripts/src_codegen/def_topi_map.py](https://github.com/awslabs/raf/blob/main/scripts/src_codegen/def_topi_map.py):

```python
OP_MAP = [
    ...
    "raf.op.tvm.softmax": "nn.softmax",
    ...
]
```

In this line, we map `raf.op.tvm.softmax` to `relay.nn.softmax`.

In addition, some Relay operators have "op strategy" registered (see [Relay Operator Strategy](https://tvm.apache.org/docs/dev/relay_op_strategy.html) for details). In short, Relay op strategy is a set of rules that determine how to lower a Relay operator. If the Relay operator has defined a strategy, we just simply register it to the RAF operator. 

For example. the dense operator `raf.op.tvm.dense` is mapped to `relay.nn.dense`, and has a strategy defined. We register this strategy to the RAF operator in [python/raf/_tvm_op/nn.py](https://github.com/awslabs/raf/blob/main/python/raf/_tvm_op/nn.py) as follows:


```python
from .._lib import strategy
_reg.register_strategy("raf.op.tvm.dense", strategy.dense_strategy)
```

#### Case 2: The operator does not have an implementation in Relay or does not have Relay op strategy

In this case, we have to write a compute and schedule function for it. We implement the compute of `softmax_dx` in [python/raf/_tvm_op/nn.py](https://github.com/awslabs/raf/blob/main/python/raf/_tvm_op/nn.py):

```python
@register_compute("raf.op.tvm.softmax_dx", level=15)
def compute_softmax_dx(attr, inputs, output_type):
    y, dy = inputs[0], inputs[1]
    axis = attr.axis
    return [(dy - _topi.sum(dy * y, axis, True)) * y]
```

and its schedule as:

```python
@generic_func
def schedule_softmax_dx(attrs, outs, target):    
    with target:
        return _topi.generic.schedule_injective(outs)


@schedule_softmax_dx.register(["cuda", "gpu"])
def schedule_softmax_dx_cuda(attrs, outs, _):
    # Schedule for softmax on CUDA devices, omitted for brevity.
    return sch

_reg.register_schedule("raf.op.tvm.softmax_dx", schedule_softmax_dx)
```

The compute function specifies the arithmetic expression of `softmax_dx`. We register the compute 
function with a `level=15`. "level" is a priority level of this dialect operator. Higher level means RAF will dispatch the base operator to this dialect operator prior to others. If not specified, the default value will be 10.

The schedule function uses the specific `schedule_softmax_dx_cuda` when the operator is executed on GPU, and uses the injective schedule in TVM for the other cases. You are welcome to craft a better schedule for the operator you implemented.

If you would like to know more about writing a compute expression, you can check this note section:

<details>

A compute expression is a "TE" expression, which stands for tensor expression in TVM. You can first check out this article to get familiar with tensor expression: [Working with Operators Using Tensor Expression](https://tvm.apache.org/docs/tutorials/get_started/tensor_expr_get_started.html#sphx-glr-tutorials-get-started-tensor-expr-get-started-py).

In addition, some commonly used functions are already implemented in TOPI so you can directly make use of them. `topi` here is similar to the standard library in C++. `_topi.nn.softmax` is a good example in the softmax's compute above.

Besides, you may also need to use `tvm.tir` in your TE compute to achieve the desired function. For example, the following TE compute represents ReLU (`y = [e if e > 0 else 0 for e in x]`):

```python
te.compute(shape, lambda *idx: tvm.tir.if_then_else(data[idx] > 0, data[idx], tvm.tir.const(0)))
```

</details>


Finally, as mentioned in the beginning of this section, one important difference between RAF op and Relay op is that RAF op does not separate arguments and attributes while Relay does. As a result, to offload a RAF op to a Relay op, we need to bridge the gap by mapping RAF arguments to the corresponding Relay arguments and attributes. The `softmax` and `softmax_dx` example is implemented in [src/op/dialect/tvm/nn.cc](https://github.com/awslabs/raf/blob/main/src/op/dialect/tvm/nn.cc):

```c++
std::vector<Value> SoftmaxSchema2Args(const SoftmaxArgs* args) {
  return {args->x};
}

std::vector<std::string> SoftmaxSchemaArgNames(const op::CallValues& call) {
  return {"x"};
}

Attrs SoftmaxSchema2Attrs(const SoftmaxArgs* args) {
  auto attrs = make_object<tvm::relay::SoftmaxAttrs>();
  attrs->axis = args->axis;
  return Attrs(attrs);
}

HashKey SoftmaxHasher(const std::vector<Type>& param_types, const Type& y_type,
                      const SoftmaxArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->axis;
  return key;
}

std::vector<Value> SoftmaxDxSchema2Args(const SoftmaxDxArgs* args) {
  return {args->y, args->dy};
}

std::vector<std::string> SoftmaxDxSchemaArgNames(const op::CallValues& call) {
  return {"y", "dy"};
}

Attrs SoftmaxDxSchema2Attrs(const SoftmaxDxArgs* args) {
  auto attrs = make_object<tvm::relay::SoftmaxAttrs>();
  attrs->axis = args->axis;
  return Attrs(attrs);
}

HashKey SoftmaxDxHasher(const std::vector<Type>& param_types, const Type& y_type,
                        const SoftmaxDxArgs* args) {
  HashKey key = GenericHasher<nullptr_t>(param_types, y_type, nullptr);
  key << args->axis;
  return key;
}
```

Here we explain the purpose of each function:
- **SoftmaxSchema2Args**: This function takes the RAF operator arguments and returns a list of Relay arguments. In the `softmax` example, RAF op has arguments `(x, axis)` while Relay op has an argument `x` and an attribute `axis`. As a result, we need to return `{x}` in this function.
- **SoftmaxSchemaArgNames**: This function returns a list of Relay argument names corresponding to `SoftmaxSchema2Args`.
- **SoftmaxSchema2Attrs**: Similar to `SoftmaxSchema2Args`, this function also takes the RAF operator arguments but returns a Relay attribute (note that the attribute will be used by the compute function we defined above). If a RAF operator does not have a Relay implementation, you have to choose either an existing Relay attribute, or define a new attribute in `src/op/dialect/tvm/attrs` and register them to [src/op/dialect/tvm/tvm_attrs.cc](https://github.com/awslabs/raf/blob/main/src/op/dialect/tvm/tvm_attrs.cc). In this example, since `softmax_dx` has the same attribute as `softmax`, we simply reuse `SoftmaxAttrs` in Relay.
- **SoftmaxHasher**: This function generates a hash key of this operator. This can avoid compiling the same operators in a model multiple times during the execution.


Finally, we register the above functions to bridge the gap between the RAF dialect op and Relay op using the `RAF_TVM` macro.

```c++
RAF_TVM(softmax, Softmax, SoftmaxArgs, SoftmaxSchema2Args, SoftmaxSchemaArgNames,
        SoftmaxSchema2Attrs, SoftmaxHasher, kCommReduce);
RAF_TVM(softmax_dx, SoftmaxDx, SoftmaxDxArgs, SoftmaxDxSchema2Args, SoftmaxDxSchemaArgNames,
        SoftmaxDxSchema2Attrs, SoftmaxDxHasher, kCommReduce);
```

where `kCommReduce` is the fusion pattern of this dialect operator. If you have no idea about determining the fusion pattern, please refer to this note:

<details>

Fusion patterns are the patterns that let the fusion pass know if this operator can be fused with others. You can check out [link](https://github.com/apache/tvm/blob/main/include/tvm/relay/op_attr_types.h) for a full list of fusion patterns and their explanations.

Note that we usually prefer to avoid `kOpaque` pattern, because it means this operator cannot fuse anything. The reason of setting an operator to `kOpaque` is usually because it is either implemented by invoking an external library (e.g., CuBLAS/CuDNN/CUDA kernel), or it has a special process in run time (e.g., dynamic shape related operators such as `shape`).

</details>

Once the implementation has been registered, we can now run this operator in RAF:

```bash
>>> raf.softmax(x, axis=0)
[[1. 1.]]
<NDArray [1 x 2] @ cpu, dtype=float32>
```

### 2.b. CuBLAS/CuDNN Dialect (GPU only)

If the operator is supported by kernel libraries such as CuBLAS or CuDNN, you should also register their implementations to achieve a better performance. In this section, we demonstrate how to register a CuDNN dialect op for `softmax`.

First, we need to register the dialect and enable it for a specific device type. In this example, we register a dialect "cudnn" and make it available on CUDA devices. Note that each dialect only needs to be registered once, so if you could find the following line in the codebase, you could skip this step. Taking "cudnn" dialect as an example, you can find its registration in [src/op/dialect/cudnn/cudnn_utils.cc](https://github.com/awslabs/raf/blob/main/src/op/dialect/cudnn/cudnn_utils.cc).

```
RAF_REGISTER_DIALECT("cudnn").set_enable(DevType::kCUDA());
```

We then demonstrate how to implement a CuDNN dialect op for `softmax`. The dialect is implemented in [src/op/dialect/cudnn/softmax.cc](https://github.com/awslabs/raf/blob/main/src/op/dialect/cudnn/softmax.cc). In particular, we derive an OpEnv, which stands for operator environment:

```c++
class SoftmaxImplementedByCUDNNSoftmaxForward : public raf::op::OpEnv {
    explicit SoftmaxImplementedByCUDNNSoftmaxForward(const CallValues& cv) {
        // Skip
    }

 public:
    ~SoftmaxImplementedByCUDNNSoftmaxForward() {
        // Skip
    }

    std::string name() const override {
      return TruncateName(GetUniqueName("raf.op.cudnn.softmax"));
    }

    void Execute(const CallValues& cv) {
      auto args = cv->args.as<raf::op::schema::SoftmaxArgs>();
      DLTensor* x = args->x;
      DLTensor* out = cv->out;
      CUDNN_CALL(cudnnSoftmaxForward(CUDNNThreadEntry::ThreadLocal()->handle, CUDNN_SOFTMAX_ACCURATE,
                                    mode, CUDNNDType(out->dtype).const_addr<1>(), xDesc, x->data,
                                    CUDNNDType(out->dtype).const_addr<0>(), yDesc, out->data));
    }

    void Execute(const std::vector<Value>& inputs, Value output) {
      CHECK_EQ(inputs.size(), 1);
      DLTensor* x = Downcast<TensorValue>(inputs[0]);
      DLTensor* out = Downcast<TensorValue>(output);
      CUDNN_CALL(cudnnSoftmaxForward(CUDNNThreadEntry::ThreadLocal()->handle, CUDNN_SOFTMAX_ACCURATE,
                                    mode, CUDNNDType(out->dtype).const_addr<1>(), xDesc, x->data,
                                    CUDNNDType(out->dtype).const_addr<0>(), yDesc, out->data));
    }

    static OpEnv* make(const CallValues& cv) {
      return new SoftmaxImplementedByCUDNNSoftmaxForward(cv);
    }
}
```

where `make` will be invoked when building the operator, and `Execute` will be invoked to execute the operator.

Finally, we register this OpEnv to be a CuDNN dialect operator:

```c++
// Now we register the dialect op "raf.op.cudnn.softmax" with level=7 to the cudnn dialect we just registered.
RAF_REGISTER_DIALECT_OP(cudnn, softmax, 7);
// Use the "make" function we just implemented to create the OpEnv for "raf.op.cudnn.softmax".
RAF_OP_ENV_MAKER("raf.op.cudnn.softmax", SoftmaxImplementedByCUDNNSoftmaxForward::make);
```

You may notice that we set the level to 7, meaning that we prefer to use the TVM dialect over cuDNN for `softmax`.

### 2.c. CUDA Dialect (GPU only)

If the operator is not supported by any kernel libraries and it cannot be scheduled well by TVM, then you probably have to write a CUDA kernel by yourself for this operator. Similar to CuDNN, we need to derive an OpEnv, but instead of simply calling CuDNN kernel, here we need to call a self-written CUDA kernel. See [src/op/dialect/cuda/embedding.cc](https://github.com/awslabs/raf/blob/main/src/op/dialect/cuda/embedding.cc) and [src/op/dialect/cuda/kernel/embedding_dx_cuda.cu](https://github.com/awslabs/raf/blob/main/src/op/dialect/cuda/kernels/embedding_dx_cuda.cu) for details.

## 3. Write Test Cases

Finally, we need to write unit tests for this operator and every of its backend implementations. You will find the unit tests for `softmax` in [tests/python/op/cudnn/test_cudnn_nn.py](https://github.com/awslabs/raf/blob/9a9bc72ae0d48686e64006f37515c196317f041f/tests/python/op/cudnn/test_cudnn_nn.py). 
Note that you should the pytest parameterize to make the test concise while covering most common cases.

```python
@with_dialect(["cudnn", "tvm"])
@pytest.mark.skipif(not raf.build.with_cuda(), reason="CUDA is not enabled")
@pytest.mark.parametrize( 
    "shape", # Test some shapes. Please pick just 2-3 representative shapes.
    [        # Please do not put too many shapes; otherwise the CI time would be too long.
        [3],
        [3, 2],
        [3, 2, 5, 8],
    ],
)
@pytest.mark.parametrize("axis", range(-4, 4))
@pytest.mark.parametrize(
    "funcs",
    [
        [raf._op.sym.softmax, torch.softmax],
        [raf._op.sym.log_softmax, torch.log_softmax],
    ],
)
def test_raf_softmax(shape, axis, funcs):
    raf_fwd, torch_fwd = funcs

    if not -len(shape) <= axis < len(shape):
        pytest.skip("axis out of range")

    class Model(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x):
            return raf_fwd(x, axis=axis)

    model = Model()

    # Random generate the input data for both RAF and PyTorch (for reference).
    m_x, t_x = randn_torch(shape, device="cuda", requires_grad=True)
    # Run the forward op (i.e., softmax) with interpreter.
    m_y = model(m_x)
    # Run the forward op (i.e., softmax) with VM.
    v_y = run_vm_model(model, "cuda", [m_x])
    # Run the forward op in PyTorch as the reference.
    t_y = torch_fwd(t_x, dim=axis)
    # Both interpreter and VM outputs should match the PyTorch output.
    check(m_y, t_y)
    check(v_y, t_y)

    # Generate the dy to test the backward op (i.e., softmax_dx).
    m_dy, t_dy = randn_torch(shape, device="cuda")
    # Run backward op in PyTorch.
    t_y.backward(t_dy)
    # Run backward op in RAF.
    m_y.backward(m_dy)
    # Check their gradients.
    check(m_x.grad, t_x.grad)
```

## Advance Supports

So far we have successfully added a new operator to RAF. 
Besides, we have some applications that need additional supports. 
Specifically, we need to define how an operator can be converted from the corresponding Relay operator to the RAF operator(if applicable), and how an operator can be casted to execute float16 data. In this section, we introduce how to provide these supports.

Please note that since an operator can be evaluated and executed without these supports, you are encouraged to file a pull request (PR) without these supports first and start work on these supports in a follow-up PR. In this way, we can control the PR size and kick off the review process sooner.

### a. Define Relay Conversion

**If the operator you have implemented is not implemented in Relay, then you can skip this section.** If the operator you implemented has a corresponding one in Relay, you need to also implement the a Relay converter. i.e., how to convert the corresponding Relay op to the RAF op. In this way, we can leverage the `FromRelay` pass to convert a Relay model with this op to RAF. Taking `softmax` as an example, we implemented a convert function in [src/op/from_relay/nn.cc](https://github.com/awslabs/raf/blob/9a9bc72ae0d48686e64006f37515c196317f041f/src/op/from_relay/nn.cc):

```c++
RAF_OP_FROM_RELAY("nn.softmax", "raf.op.softmax",                                    \
    [&](const Attrs& attrs, const Array<Expr>& args, const VarValueMap& val_map) {   \
        Array<Expr> raf_args = args;                                                 \
        const auto* relay_attrs = attrs.as<SoftmaxAttrs>();                          \
        raf_args.push_back(MakeConstant(ScalarValue::make(relay_attrs->axis)));      \
        return raf_args;                                                             \
    })
```

The first argument (`nn.softmax`) is the Relay op name; the second argument (`raf.op.softmax`) is the RAF op name. The third argument is a converter function, and its purpose is to map Relay arguments and attributes to the RAF arguments. In the case of `softmax`, we have:

* Relay
    * Arguments: x.
    * SoftmaxAttrs: axis.
* RAF
    * SoftmaxArgs: x, axis.

Accordingly, we create a converter returning `raf_args` that aligns to `SoftmaxArgs` in RAF.

### b. Define Auto Casting Rules

Automatic mixed precision (AMP) is getting popular in recent years. Its core idea is training or inference the model with some operators running with full precision data while others running in half precision to achieve better performance. In RAF, we let every operator specify the auto casting rules to indicate whether the operator can be executed with half precision data. The casting rules are called "type hint", and are implemented in [python/raf/amp/type_hint.py](https://github.com/awslabs/raf/blob/9a9bc72ae0d48686e64006f37515c196317f041f/python/raf/amp/type_hints.py).

In general, there are 3 types of casting rules:

1. **Always Cast**: The operator must be running with AMP type data when possible to achieve the best performance. For example, the most time-consuming operators such as `Conv2D` and `MatMul` fall into this category.
2. **Never Cast**: Some operators does not support AMP type and can only work with full precision types. For example, `softmax`, `erf`, `nll_loss` are never cast operators.
3. **Infer Cast**: The last category of operators can run with either full precision or AMP type data. In this case, we should check its argument to determine which one to run in order to minimize the casting overheads. For example, `max_pool2d`, `relu` are infer cast operators.

You need to check the underlying implementations of the operator to determine the casting type. Basically, if the operator is compute intensive and float16 can achieve 2x or more speedup compared to float32, then it should always cast. If the operator can only take float32 data due to its computation and accuracy limitation, then it should never be casted. Finally, if the operator can run both float32 and float16 with similar performance, then if should be infer cast. In this case, we attempt to minimize the number of cast ops, so we try to follow the dtype of its arguments. In addition, you could refer to [Op-Specific Behavior](https://pytorch.org/docs/stable/amp.html#op-specific-behavior) for Pytorch's autocast rules.

In `type_hint.py`, we provide utilities for the above 3 casting rules, so you can just add one line to `type_hint.py` if the operator is in one of the 3 categories. Taking `softmax` as an example, we added:

```python
register_op_cast_rule("raf.op.softmax", generic_cast(False, 1))
```

where `False` means we never cast this operator, and `1` means the first argument of this operator is a data argument instead of an attribute argument. Since it is usually illegal to cast attribute arguments (they are usually constants like eps or axis), we have to explicitly tell AutoCast to only focus on the first `N` arguments.

## Summary

In summary, here is the checklist you can refer to when adding a new operator. If any of them is not applicable to your case, you can simply check and ignore them.

- [ ] Define schema
- [ ] Define operator symbol
- [ ] Declare the behavior
- [ ] Define type function
- [ ] Define gradient function
- [ ] Define FromRelay conversion
- [ ] Define auto casting rules
- [ ] Implement dialect ops
  - [ ] Implement TVM/Relay dialect
  - [ ] Implement CuBLAS/CuDNN dialect
  - [ ] Implement CUTLASS dialect
  - [ ] Implement CUDA kernel dialect
- [ ] Write a unit test
  - [ ] Test forward
      - [ ] Test type function
      - [ ] Test Relay conversion
      - [ ] Test auto casting
      - [ ] Test gradient
      - [ ] Test VM
  - [ ] Test backward
      - [ ] Test VM
