"""
The tuning utilities.
"""
import time

import numpy as np
import tvm

import mnm
from mnm._core.executor import VMExecutor
from mnm._core.module import Module
from mnm.testing import randn, run_infer_type
from tvm import auto_scheduler, autotvm
from tvm.auto_scheduler import compute_dag


def run_fusion(model, args):
    """Run the fusion pass for the given model"""
    # pylint: disable=protected-access
    func = model._internal(*args).func
    func = run_infer_type(func)
    func = mnm._ffi.pass_.FuseOps(func, 3)
    func = run_infer_type(func)
    return func


def extract_tuning_tasks(func, target, args):
    """Extract tuning tasks from the given function and the target"""
    old_autotvm_silent = autotvm.GLOBAL_SCOPE.silent
    autotvm.GLOBAL_SCOPE.silent = True

    env = auto_scheduler.relay_integration.TracingEnvironment(
        auto_scheduler.relay_integration.TracingMode.EXTRACT_COMPLEX_TASK_ONLY
    )
    with env:
        # TODO(comaniac): Whether to make a new thread?
        mod = Module()
        mod[tvm.ir.GlobalVar("main")] = func
        executor = VMExecutor(mod, target)
        executor.make_executor()(*args)

    autotvm.GLOBAL_SCOPE.silent = old_autotvm_silent

    tvm_target = tvm.target.Target(target)

    tasks = []
    weights = []
    for wkl_key, weight in env.wkl_key_to_weight.items():
        tasks.append(
            auto_scheduler.SearchTask(
                workload_key=wkl_key,
                target=tvm_target,
                target_host=None,
                hardware_params=None,
                # When auto scheduler is used in end to end network, try to apply layout rewrite
                # to improve the overall performance
                layout_rewrite_option=compute_dag.LayoutRewriteOption.get_target_default(
                    tvm_target, True
                ),
            )
        )
        weights.append(weight)
    return tasks, weights


def run_tuning(func, device, args, log_file):
    """Tune the given tasks"""

    print("Extracting tasks...")
    tasks, weights = extract_tuning_tasks(func, device, args)
    for idx, (task, weight) in enumerate(zip(tasks, weights)):
        print("=== Task %d (weight %d) ===" % (idx, weight))
        print(task.compute_dag)

    print("Tuning...")

    measure_device = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=400, timeout=10)

    tuner = auto_scheduler.TaskScheduler(tasks, weights)

    # The total trials for tuning all tasks. Here we apply a simple equation
    # to determine the trails.
    n_trials = 32 * min(len(tasks), 10)

    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=n_trials,
        runner=measure_device.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )
    tuner.tune(tune_option)
    del measure_device
    print("Done tuning")


def profile_vm_func_with_schedule(func, device, args, log_file=None):
    """Helper function to execute model with VM with tuned schedules"""
    mod = Module()
    mod[tvm.ir.GlobalVar("main")] = func

    # Get rid of the first run because it includes the compilation.
    executor = VMExecutor(mod, device)
    executor.make_executor(log_file)(*args)

    # Run N times to profile the latency.
    latencies = []
    for _ in range(10):
        start = time.time()
        out = executor.make_executor(log_file)(*args)
        out.asnumpy()
        latencies.append(time.time() - start)
    print("Latency (ms): %.2f ms" % (np.median(latencies) * 1000))

    return out


def run_test(device):
    """Run the test script"""
    # pylint: disable=no-self-use, invalid-name

    n, ci, h, w, co = 8, 16, 56, 56, 16

    class Model(mnm.Model):
        """Model to be tested."""

        def build(self):
            """Do nothing"""

        @mnm.model.trace
        def forward(self, data, weight):
            """Model definition. It includes 3 conv2d and 2 of them are identical,
            so we should extract 2 tuning tasks.
            """
            out = mnm.conv2d(data, weight, padding=(1, 1), layout="NCHW", kernel_layout="OIHW")
            out = mnm.relu(out)
            out = mnm.conv2d(out, weight, padding=(1, 1), layout="NCHW", kernel_layout="OIHW")
            out = mnm.relu(out)
            out = mnm.conv2d(out, weight, padding=(0, 0), layout="NCHW", kernel_layout="OIHW")
            out = mnm.relu(out)
            return out

    model = Model()
    model.infer_mode()

    m_x, _ = randn((n, ci, h, w), device=device)
    m_w, _ = randn((co, ci, 3, 3), device=device)
    func = run_fusion(model, [m_x, m_w])

    # Profile with the fallback schedules (~1.15 ms in this example on g4dn).
    profile_vm_func_with_schedule(func, device, [m_x, m_w])

    # Tuning
    log_file = "tuning.json"
    run_tuning(func, device, [m_x, m_w], log_file)

    # Profile with the tuned schedules (~0.99 ms in this example on g4dn after 256 trials).
    profile_vm_func_with_schedule(func, device, [m_x, m_w], log_file)


if __name__ == "__main__":
    run_test("cuda")
