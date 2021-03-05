"""
The tuning utilities.
"""
# pylint: disable=too-many-locals
import os
import time

import numpy as np
import tvm

import mnm
from mnm._core.executor import VMExecutor
from mnm._core.module import Module
from mnm.model.trace import _get_func_inputs
from mnm.testing import randn
from mnm.testing.utils import ir_fusion
from tvm import auto_scheduler, autotvm
from tvm.auto_scheduler import compute_dag
from tvm.auto_scheduler.measure import MeasureErrorNo
from tvm.auto_scheduler.measure_record import RecordReader


def extract_tuning_tasks(func, device, args):
    """Extract tuning tasks from the given function and the target.

    Parameters
    ----------
    func: relay.Function
        The function to be extracted.

    device: str
        The target device.

    args: List[mnm.ndarray]
        A list of input arguments.

    Returns
    -------
    task_n_weights: Tuple[List[SearchTask], List[int]]
        A tuple of tasks and weights (appearance in the model).
    """
    old_autotvm_silent = autotvm.GLOBAL_SCOPE.silent
    autotvm.GLOBAL_SCOPE.silent = True

    env_tracing_task = auto_scheduler.relay_integration.TracingEnvironment(
        auto_scheduler.relay_integration.TracingMode.EXTRACT_COMPLEX_TASK_ONLY
    )
    with env_tracing_task:
        # TODO(comaniac): Whether to make a new thread?
        mod = Module()
        mod[tvm.ir.GlobalVar("main")] = func
        executor = VMExecutor(mod, device)
        with tvm.transform.PassContext(
                config={"relay.backend.use_auto_scheduler": True,
                        "mnm.tvmjit.extract_task": True},
                disabled_pass={"AutoSchedulerLayoutRewrite"},
        ):
            executor.vm.run(*args)

    autotvm.GLOBAL_SCOPE.silent = old_autotvm_silent

    tvm_target = tvm.target.Target(device)

    tasks = []
    weights = []
    for wkl_key, weight in env_tracing_task.wkl_key_to_weight.items():
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


def run_tuning(model, device, args, log_file, *, optimize=None,
               n_trials=lambda l: 300 * min(l, 100), only_tune_new_tasks=False):
    """Tune the given tasks.

    Parameters
    ----------
    model: mnm.BaseModel
        The model to be tuned.

    device: str
        The target device.

    log_file: str
        The log file to dump the tuning records. If the file already contains tuning records,
        we use them to initialize the task scheduler and new records will be appended.

    optimize: Optional[Callable[[relay.Function], relay.Function]]
        An optimization function for Meta Relay functions.

    n_trials: Callable[[int], int] or int
        An integer of total number of measurement trials, or a function that determines
        the total number of measurement trials by taking the task number.
        Default is at maximum 30k = 300 * min(100, task_num).

    only_tune_new_tasks: bool
        If True, then we only tune the tasks that have no valid records in the given
        log file.

    args: List[mnm.ndarray]
        A list of input arguments.
    """
    # pylint: disable=protected-access

    record = model._internal(*args)
    func = record.func
    if optimize:
        func = optimize(func)
    inputs = _get_func_inputs(record, args, {}, get_handle=False)

    print("Extracting tasks...")
    tasks, weights = extract_tuning_tasks(func, device, inputs)
    ori_task_num = len(tasks)

    if only_tune_new_tasks and os.path.exists(log_file):
        print("Remove tasks that have valid records in %s" % log_file)
        # Collect workload keys in the log file.
        workload_keys = set()
        tvm_target = tvm.target.Target(device)
        for inp, res in RecordReader(log_file):
            if res.error_no != MeasureErrorNo.NO_ERROR:
                continue
            if inp.task.target.kind.name != tvm_target.kind.name:
                continue
            workload_keys.add(inp.task.workload_key)

        # Erase tasks that already have valid records in the log file.
        keep_tasks = []
        keep_weights = []
        for task, weight in zip(tasks, weights):
            if task.workload_key not in workload_keys:
                keep_tasks.append(task)
                keep_weights.append(weight)
        tasks, weights = keep_tasks, keep_weights

    for idx, (task, weight) in enumerate(zip(tasks, weights)):
        print("=== Task %d: %s (weight %d) ===" % (idx, task.workload_key, weight))
        print(task.compute_dag)

    print("Tuning %d out of %s tasks..." % (len(tasks), ori_task_num))

    measure_device = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=400, timeout=10)

    if os.path.exists(log_file):
        tuner = auto_scheduler.TaskScheduler(tasks, weights, load_log_file=log_file)
    else:
        tuner = auto_scheduler.TaskScheduler(tasks, weights)

    if callable(n_trials):
        n_trials = n_trials(len(tasks))

    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=n_trials,
        runner=measure_device.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )
    tuner.tune(tune_option)
    del measure_device
    print("Done tuning. Records saved in %s" % log_file)


def profile_vm_func_with_schedule(expr, device, args, log_file=None):
    """Helper function to execute model with VM with tuned schedules"""
    if isinstance(expr, Module):
        mod = expr
    else:
        mod = Module()
        mod[tvm.ir.GlobalVar("main")] = expr

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
    # pylint: disable=no-self-use, invalid-name, protected-access

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
    mod = model._internal(m_x, m_w).mod
    mod = ir_fusion(mod)

    # Profile with the fallback schedules (~1.15 ms in this example on g4dn).
    profile_vm_func_with_schedule(mod, device, [m_x, m_w])

    # Tuning
    log_file = "tuning.json"
    run_tuning(mod['main'], device, [m_x, m_w], log_file, n_trials=128)

    # Profile with the tuned schedules (~0.99 ms in this example on g4dn after 256 trials).
    profile_vm_func_with_schedule(mod, device, [m_x, m_w], log_file)


if __name__ == "__main__":
    run_test("cuda")
