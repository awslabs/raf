"""
The tuning utilities.
"""
# pylint: disable=too-many-locals, too-many-arguments
# pylint: disable=missing-class-docstring, missing-function-docstring, no-self-use
import os
from copy import copy

import tvm

import mnm
from mnm._core.executor import MetaFallbackContext, VMExecutor
from mnm.model.trace import _get_func_inputs
from mnm.testing import randn
from tvm import auto_scheduler, autotvm
from tvm.auto_scheduler import compute_dag
from tvm.auto_scheduler.measure import MeasureErrorNo
from tvm.auto_scheduler.measure_record import RecordReader


def extract_tuning_tasks(mod, args, device, *, optimize=None):
    """Extract tuning tasks from the given function and the target.

    Parameters
    ----------
    mod: mnm.Model or IRModule
        The module to be extracted.

    args: List[mnm.ndarray]
        A list of input arguments.

    device: str
        The target device.

    optimize: Optional[Callable[[relay.Function], relay.Function]]
        An optimization function for Meta Relay functions.

    Returns
    -------
    task_n_weights: Tuple[List[SearchTask], List[int]]
        A tuple of tasks and weights (appearance in the model).
    """
    # pylint: disable=protected-access
    if isinstance(mod, mnm.Model):
        record = mod._internal(*args)
        mod = record.mod
        if optimize is not None:
            mod = optimize(mod)
        args = _get_func_inputs(record, args, {}, get_handle=False)

    old_auto_scheduler_fallback_context = auto_scheduler.DispatchContext.current
    auto_scheduler.DispatchContext.current = MetaFallbackContext(verbose=0)
    old_autotvm_silent = autotvm.GLOBAL_SCOPE.silent
    autotvm.GLOBAL_SCOPE.silent = True

    env_tracing_task = auto_scheduler.relay_integration.TracingEnvironment(
        auto_scheduler.relay_integration.TracingMode.EXTRACT_COMPLEX_TASK_ONLY
    )
    with env_tracing_task:
        # TODO(comaniac): Whether to make a new thread?
        executor = VMExecutor(mod, device)
        with tvm.transform.PassContext(
                config={"relay.backend.use_auto_scheduler": True,
                        "mnm.tvmjit.extract_task": True},
                disabled_pass={"AutoSchedulerLayoutRewrite"},
        ):
            executor.vm.run(*args)

    autotvm.GLOBAL_SCOPE.silent = old_autotvm_silent
    auto_scheduler.DispatchContext.current = old_auto_scheduler_fallback_context

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

def tune_tasks(tasks, weights, log_file, n_trials):
    """Tune a set of given tasks.

    tasks: List[tvm.auto_scheduler.SearchTask]
        The list of tasks.

    weights: List[int]
        The weight of each task.

    log_file: str
        The path to log file for storing tuning logs.

    n_trials: Callable[[int], int] or int
        An integer of total number of measurement trials, or a function that determines
        the total number of measurement trials by taking the task number.
    """
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

def run_tuning(model, device, args, log_file, *, optimize=None,
               n_trials=lambda l: 300 * min(l, 100), only_tune_new_tasks=False):
    """Tune the given tasks.

    Parameters
    ----------
    model: mnm.BaseModel
        The model to be tuned.

    device: str
        The target device.

    args: List[mnm.ndarray]
        A list of input arguments.

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
    """
    print("Extracting tasks...")
    tasks, weights = extract_tuning_tasks(model, args, device, optimize=optimize)
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
    tune_tasks(tasks, weights, log_file, n_trials)

def tune_op(sch_file, model_cls, gen_arg_func, space_dict, n_trials_per_task=128, device="cuda"):
    """A helper function to construct and tune tasks for an op.

    sch_file: str
        The tuning log.

    model_cls: mnm.Model
        The Meta model that includes only one op.

    gen_arg_func: Callable
        The function to generate model arguments from a config.

    space_dict: Dict[str, List[Any]]
        A map from the arguments of gen_arg_func to possible values.

    n_trials_per_task: int
        Trials for each generated task. Default 128.

    device: str
        The target device. Default cuda.
    """
    configs = []
    space_list = list(space_dict.items())

    def build_space_dfs(key_idx, curr_cfg):
        if key_idx == len(space_list): # Leaf
            configs.append(copy(curr_cfg))
            return

        key_name, vals = space_list[key_idx]
        for val in vals:
            curr_cfg[key_name] = val
            build_space_dfs(key_idx + 1, curr_cfg)
            del curr_cfg[key_name]

    build_space_dfs(0, {})
    args = [gen_arg_func(**config) for config in configs]

    print("Constructing %d tuning tasks..." % len(args))
    tasks = []
    for args in args:
        m_model = model_cls()
        m_model.infer_mode()
        m_model.to(device=device)

        extract_tasks, _ = extract_tuning_tasks(m_model, args, device)
        assert len(extract_tasks) == 1
        tasks.append(extract_tasks[0])

    tune_tasks(tasks, [1 for _ in range(len(tasks))], sch_file,
               n_trials=n_trials_per_task * len(tasks))


def tune_softmax_dx(sch_file, space_dict=None, device="cuda"):
    """Tune softmax_dx with various shapes.

    sch_file: str
        The tuning log.

    space_dict: Optional[Dict[str, List[Any]]]
        The target space (configs) of this op. If not present, the default space will be used.

    device: str
        The target device. Default cuda.
    """
    class SoftmaxDxModel(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x, y, dy):
            return mnm.softmax_dx(x, y, dy)

    def gen_arg_func(batch_size, seq_length):
        shape = [batch_size, 12, seq_length, seq_length]
        m_x, _ = randn(shape, device=device)
        m_y, _ = randn(shape, device=device)
        m_dy, _ = randn(shape, device=device)
        return [m_x, m_y, m_dy]

    if space_dict is None:
        space_dict = {
            "batch_size": [1, 2, 3, 4, 5, 7, 11, 13, 16, 17, 19, 23, 29, 31, 32],
            "seq_length": [5, 32, 64, 128]
        }

    tune_op(sch_file, SoftmaxDxModel, gen_arg_func, space_dict, device=device)


def tune_layer_norm(sch_file, space_dict=None, device="cuda"):
    """Tune layer_norm with various shapes.

    sch_file: str
        The tuning log.

    space_dict: Optional[Dict[str, List[Any]]]
        The target space (configs) of this op. If not present, the default space will be used.

    device: str
        The target device. Default cuda.
    """

    class LayerNormModel(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x, scale, bias):
            return mnm.layer_norm(x, scale, bias, eps=1e-12)

    def gen_arg_func(batch_size, seq_length, hidden_size):
        shape = [batch_size, seq_length, hidden_size]
        m_x, _ = randn(shape, device=device)
        m_scale, _ = randn((shape[2],), device=device)
        m_bias, _ = randn((shape[2],), device=device)
        return [m_x, m_scale, m_bias]

    if space_dict is None:
        space_dict = {
            "batch_sizes": [1, 2, 3, 4, 5, 7, 11, 13, 16, 17, 19, 23, 29, 31, 32],
            "seq_lengths": [5, 32, 64, 128],
            "hidden_sizes": [512, 768, 1024]
        }

    tune_op(sch_file, LayerNormModel, gen_arg_func, space_dict, device=device)


def tune_layer_norm_dx(sch_file, space_dict=None, device="cuda"):
    """Tune layer_norm_dx with various shapes.

    sch_file: str
        The tuning log.

    space_dict: Optional[Dict[str, List[Any]]]
        The target space (configs) of this op. If not present, the default space will be used.

    device: str
        The target device. Default cuda.
    """

    class LayerNormDxModel(mnm.Model):
        def build(self):
            pass

        @mnm.model.trace
        def forward(self, x, scale, dy):
            return mnm.layer_norm_dx(x, scale, dy, eps=1e-12)

    def gen_arg_func(batch_size, seq_length, hidden_size):
        shape = [batch_size, seq_length, hidden_size]
        m_x, _ = randn(shape, device=device)
        m_scale, _ = randn((shape[2],), device=device)
        m_dy, _ = randn(shape, device=device)
        return [m_x, m_scale, m_dy]

    if space_dict is None:
        space_dict = {
            "batch_sizes": [1, 2, 3, 4, 5, 7, 11, 13, 16, 17, 19, 23, 29, 31, 32],
            "seq_lengths": [5, 32, 64, 128],
            "hidden_sizes": [512, 768, 1024]
        }

    tune_op(sch_file, LayerNormDxModel, gen_arg_func, space_dict, device=device)


def tune_all_ops(sch_file, device="cuda"):
    """Tune all listed ops.

    sch_file: str
        The tuning log.

    device: str
        The target device. Default cuda.
    """
    for tune_func in [tune_softmax_dx, tune_layer_norm, tune_layer_norm_dx]:
        tune_func(sch_file, device=device)
