#include <gtest/gtest.h>

#include <mnm/device.h>
#include <mnm/memory_pool.h>
#include <mnm/device.h>
#include <mnm/op.h>
#include <mnm/ir.h>
#include <tvm/runtime/container/array.h>
#include <mnm/op_profiler.h>

using mnm::Device;
using mnm::DevType;
using mnm::Op;
using tvm::runtime::Array;
using namespace mnm::ir;
using namespace mnm::op_profiler;

#ifdef MNM_USE_CUDA
TEST(OpProfilerGPU, TestNoCall) {
  Device dev{DevType::kCUDA(), 0};

  CUDAOpProfiler* profiler = CUDAOpProfiler::Make(dev);

  Var a("a", TensorType::Scalar(DataType::Float(32)));
  Var b("b", TensorType::Scalar(DataType::Float(32)));
  Array<Expr> vars;
  vars.push_back(a);
  vars.push_back(b);
  Tuple t = Tuple(vars);
  auto not_call = TupleGetItem(t, 1);
  float latency = profiler->ProfileOp(not_call);
  LOG(INFO) << "Latency is: " << latency;
  ASSERT_EQ(latency, 0.0f);
}

// We rely on the internal dispatch mechanism to dispatch ops to dialects

TEST(OpProfilerGPU, TestOp) {
  Device dev{DevType::kCUDA(), 0};

  CUDAOpProfiler* profiler = CUDAOpProfiler::Make(dev);

  Op op = Op::Get("mnm.op.matmul_nt");
  Array<PrimExpr> in_shape0;
  in_shape0.push_back(64);
  in_shape0.push_back(32);
  Var arg0("arg0", TensorType(in_shape0, DataType::Float(32)));
  arg0->checked_type_ = TensorType(in_shape0, DataType::Float(32));

  Array<PrimExpr> in_shape1;
  in_shape1.push_back(32);
  in_shape1.push_back(32);
  Var arg1("arg1", TensorType(in_shape1, DataType::Float(32)));
  arg1->checked_type_ = TensorType(in_shape1, DataType::Float(32));

  Array<Expr> args;
  args.push_back(arg0);
  args.push_back(arg1);

  auto fake_call = Call(op, args);
  Array<PrimExpr> ret_shape;
  ret_shape.push_back(32);
  ret_shape.push_back(32);
  fake_call->checked_type_ = TensorType(ret_shape, DataType::Float(32));

  float latency = profiler->ProfileOp(fake_call);
  LOG(INFO) << "Latency is: " << latency;
  ASSERT_GT(latency, 0.0f);

  // The returned latency should be exactly the same as the previous one due to cache hit.
  float latency2 = profiler->ProfileOp(fake_call);
  LOG(INFO) << "Latency is: " << latency2;
  ASSERT_FLOAT_EQ(latency, latency2);
}
#endif

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
