# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
The tuning utilities.
"""
# pylint: disable=too-many-locals, too-many-arguments, protected-access
# pylint: disable=missing-class-docstring, missing-function-docstring, no-self-use
# pylint: disable=attribute-defined-outside-init
import os
from copy import copy

import tvm

import raf
from raf._core.executor import MetaFallbackContext
from raf._core.ndarray import array
from raf._core.executor import VMExecutor
from raf.model.trace import _get_func_inputs
from raf.testing import randn, randint, randn_torch
from tvm import auto_scheduler, autotvm
from tvm.auto_scheduler import compute_dag


def extract_tuning_tasks(mod_or_executor, args, device, *, fusion=False, pass_seq=None):
    """Extract tuning tasks from the given function and the target.

    Parameters
    ----------
    mod_or_executor: Union[raf.Model, raf.ir.IRModule]
        The module or the compiled VMExecutor to be extracted.

    args: List[raf.ndarray]
        A list of input arguments.

    device: str
        The target device.

    fusion: bool
        Whether to apply fusion. Default: False.

    pass_seq: Optional[RAFSequential]
        A pass sequence to be applied.

    Returns
    -------
    task_n_weights: Tuple[List[SearchTask], List[int]]
        A tuple of tasks and weights (appearance in the model).
    """
    # pylint: disable=protected-access
    if isinstance(mod_or_executor, VMExecutor):
        executor = mod_or_executor
    else:
        assert isinstance(mod_or_executor, (raf.Model, raf.ir.IRModule))
        if isinstance(mod_or_executor, raf.Model):
            record = mod_or_executor._internal(*args)
            mod = record.mod
            if pass_seq is not None:
                mod = pass_seq(mod)
            args = _get_func_inputs(record, args, {}, get_handle=False)
        else:  # raf.ir.IRModule
            mod = mod_or_executor
        disabled_pass = ["FuseDialect", "FuseTVM"] if not fusion else []
        disabled_pass.append("AutoSchedulerLayoutRewrite")
        with raf.ir.PassContext(opt_level=3, disabled_pass=disabled_pass):
            # Use dry run to skip the op execution; otherwise it becomes a chicken-egg problem:
            # we need to run through every ops to collect tuning tasks, but we cannot execute ops
            # without schedules.
            executor = VMExecutor(mod, device, dryrun=True)

    old_auto_scheduler_fallback_context = auto_scheduler.DispatchContext.current
    auto_scheduler.DispatchContext.current = MetaFallbackContext(verbose=0)
    old_autotvm_silent = autotvm.GLOBAL_SCOPE.silent
    autotvm.GLOBAL_SCOPE.silent = True

    env_tracing_task = auto_scheduler.relay_integration.TracingEnvironment(
        auto_scheduler.relay_integration.TracingMode.EXTRACT_COMPLEX_TASK_ONLY
    )
    with env_tracing_task:
        with raf.ir.PassContext(
            config={"relay.backend.use_auto_scheduler": True, "raf.tvm.allow_jit_failure": True},
            disabled_pass={"AutoSchedulerLayoutRewrite"},
        ):
            executor.vm.run(*args)

    autotvm.GLOBAL_SCOPE.silent = old_autotvm_silent
    auto_scheduler.DispatchContext.current = old_auto_scheduler_fallback_context

    tvm_target = tvm.target.Target(device)

    tasks = []
    weights = []
    for wkl_key, (weight, func_names) in env_tracing_task.wkl_key_to_weight.items():
        tasks.append(
            auto_scheduler.SearchTask(
                workload_key=wkl_key,
                target=tvm_target,
                hardware_params=None,
                # When auto scheduler is used in end to end network, try to apply layout rewrite
                # to improve the overall performance
                layout_rewrite_option=compute_dag.LayoutRewriteOption.get_target_default(
                    tvm_target, True
                ),
                desc=",".join(func_names),
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
    measure_device = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=400, timeout=20)

    # FIXME(comaniac): Remove this custom objective function after
    # https://github.com/apache/tvm/pull/8984
    def weighted_sum(costs):
        score = sum(c * w for c, w in zip(costs, weights))
        if not hasattr(score, "value"):
            return tvm.tir.expr.FloatImm("float32", score)
        return score

    if os.path.exists(log_file):
        tuner = auto_scheduler.TaskScheduler(
            tasks, weights, load_log_file=log_file, objective_func=weighted_sum
        )
    else:
        tuner = auto_scheduler.TaskScheduler(tasks, weights, objective_func=weighted_sum)

    if callable(n_trials):
        n_trials = n_trials(len(tasks))

    print("Start tuning for maximal %d trials" % n_trials)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=n_trials,
        runner=measure_device.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )
    tuner.tune(tune_option)
    del measure_device
    print("Done tuning. Records saved in %s" % log_file)


def run_tuning(
    model_or_executor,
    device,
    args,
    log_file,
    *,
    fusion=False,
    pass_seq=None,
    n_trials=lambda l: 300 * min(l, 100),
    only_tune_tasks_with_name=None,
    only_extract_tasks=False
):
    """Tune the given tasks.

    Parameters
    ----------
    model_or_executor: Union[raf.BaseModel, VMProfilerExecutor]
        The model to be tuned, or the compiled VMProfilerExecutor.

    device: str
        The target device.

    args: List[raf.ndarray]
        A list of input arguments.

    log_file: str
        The log file to dump the tuning records. If the file already contains tuning records,
        we use them to initialize the task scheduler and new records will be appended.

    fusion: bool
        Whether to apply fusion. Default: False.

    pass_seq: Optional[RAFSequential]
        A pass sequence to be applied.

    n_trials: Callable[[int], int] or int
        An integer of total number of measurement trials, or a function that determines
        the total number of measurement trials by taking the task number.
        Default is at maximum 30k = 300 * min(100, task_num).

    only_tune_tasks_with_name: Optional[List[str]]
        When specify with a list of name tokens, only the tasks with the tokens in their names
        will be tuned.

    only_extract_tasks: bool
        Whether to extract and print tasks only without actual tuning them.
    """
    print("Extracting tasks...")
    tasks, weights = extract_tuning_tasks(
        model_or_executor, args, device, fusion=fusion, pass_seq=pass_seq
    )
    ori_task_num = len(tasks)

    if only_tune_tasks_with_name is not None:
        only_tune_tasks_with_name = [token.strip() for token in only_tune_tasks_with_name]
        print("Selecting tasks with specified names: %s" % ",".join(only_tune_tasks_with_name))
        keep_tasks = []
        keep_weights = []
        for task, weight in zip(tasks, weights):
            if any([task.desc.find(token) != -1 for token in only_tune_tasks_with_name]):
                keep_tasks.append(task)
                keep_weights.append(weight)
        tasks, weights = keep_tasks, keep_weights

    for idx, (task, weight) in enumerate(zip(tasks, weights)):
        print("=== Task %d: %s (weight %d) ===" % (idx, task.workload_key, weight))
        print("Function: %s" % task.desc)
        print(task.compute_dag)

    if only_extract_tasks:
        return

    print("Tuning %d out of %s tasks..." % (len(tasks), ori_task_num))
    tune_tasks(tasks, weights, log_file, n_trials)


def tune_op(
    sch_file,
    model_cls,
    gen_arg_func,
    space_dict,
    n_trials=None,
    device="cuda",
    fusion=False,
    only_extract_tasks=False,
):
    """A helper function to construct and tune tasks for an op.

    sch_file: str
        The tuning log.

    model_cls: raf.Model
        The RAF model that includes only one op.

    gen_arg_func: Callable
        The function to generate model arguments from a config.

    space_dict: Dict[str, List[Any]]
        A map from the arguments of gen_arg_func to possible values.

    n_trials: Optional[int]
        Tuning trials. If None, we use #task * 1.2 * 64 trials.

    device: str
        The target device. Default cuda.

    fusion: bool
        Whether to apply fusion.

    only_extract_tasks: bool
        Whether to extract and print tasks only without actual tuning them.
    """
    configs = []
    space_list = list(space_dict.items())

    def build_space_dfs(key_idx, curr_cfg):
        if key_idx == len(space_list):  # Leaf
            configs.append(copy(curr_cfg))
            return

        key_name, vals = space_list[key_idx]
        for val in vals:
            curr_cfg[key_name] = val
            build_space_dfs(key_idx + 1, curr_cfg)
            del curr_cfg[key_name]

    build_space_dfs(0, {})
    args = [gen_arg_func(**config) for config in configs]

    print("Constructing %d tuning tasks for %s ..." % (len(args), model_cls.__name__))
    tasks = []
    for model_args, input_args in args:
        m_model = model_cls(*model_args)
        m_model.infer_mode()
        m_model.to(device=device)

        extract_tasks, _ = extract_tuning_tasks(
            m_model, input_args, device, fusion=fusion, pass_seq=None
        )
        assert len(extract_tasks) == 1
        tasks.append(extract_tasks[0])

    for idx, task in enumerate(tasks):
        print("=== Task %d: %s ===" % (idx, task.workload_key))
        print("Function: %s" % task.desc)

    if only_extract_tasks:
        return

    n_trials = n_trials if n_trials is not None else int(len(tasks) * 1.2 * 64)
    tune_tasks(tasks, [1 for _ in range(len(tasks))], sch_file, n_trials=n_trials)


def tune_softmax_dx(sch_file, space_dict=None, device="cuda", only_extract_tasks=False):
    """Tune softmax_dx with various shapes.

    sch_file: str
        The tuning log.

    space_dict: Optional[Dict[str, List[Any]]]
        The target space (configs) of this op. If not present, the default space will be used.

    device: str
        The target device. Default cuda.

    only_extract_tasks: bool
        Whether to extract and print tasks only without actual tuning them.
    """

    class SoftmaxDxModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, y, dy):
            return raf.softmax_dx(x, y, dy)

    def gen_arg_func(batch_size, seq_length):
        shape = [batch_size, 12, seq_length, seq_length]
        m_x, _ = randn(shape, device=device)
        m_y, _ = randn(shape, device=device)
        m_dy, _ = randn(shape, device=device)
        return [], [m_x, m_y, m_dy]

    if space_dict is None:
        space_dict = {"batch_size": [1, 4, 8, 16, 32, 64], "seq_length": [5, 32, 64, 128]}

    tune_op(
        sch_file,
        SoftmaxDxModel,
        gen_arg_func,
        space_dict,
        device=device,
        only_extract_tasks=only_extract_tasks,
    )


def tune_layer_norm(sch_file, space_dict=None, device="cuda", only_extract_tasks=False):
    """Tune layer_norm with various shapes.

    sch_file: str
        The tuning log.

    space_dict: Optional[Dict[str, List[Any]]]
        The target space (configs) of this op. If not present, the default space will be used.

    device: str
        The target device. Default cuda.

    only_extract_tasks: bool
        Whether to extract and print tasks only without actual tuning them.
    """

    class LayerNormModel(raf.Model):
        def build(self, eps):
            self.eps = eps

        @raf.model.trace
        def forward(self, x, scale, bias):
            return raf.layer_norm(x, scale, bias, eps=self.eps)

    def gen_arg_func(batch_size, seq_length, hidden_size, eps):
        shape = [batch_size, seq_length, hidden_size]
        m_x, _ = randn(shape, device=device)
        m_scale, _ = randn((shape[2],), device=device)
        m_bias, _ = randn((shape[2],), device=device)
        return [eps], [m_x, m_scale, m_bias]

    if space_dict is None:
        space_dict = {
            "batch_size": [1, 4, 8, 16, 32, 64],
            "seq_length": [128],
            "hidden_size": [768],
            "eps": [1e-5, 1e-12],
        }

    tune_op(
        sch_file,
        LayerNormModel,
        gen_arg_func,
        space_dict,
        device=device,
        only_extract_tasks=only_extract_tasks,
    )


def tune_layer_norm_dx(sch_file, space_dict=None, device="cuda", only_extract_tasks=False):
    """Tune layer_norm_dx with various shapes.

    sch_file: str
        The tuning log.

    space_dict: Optional[Dict[str, List[Any]]]
        The target space (configs) of this op. If not present, the default space will be used.

    device: str
        The target device. Default cuda.

    only_extract_tasks: bool
        Whether to extract and print tasks only without actual tuning them.
    """

    class LayerNormDxModel(raf.Model):
        def build(self, eps):
            self.eps = eps

        @raf.model.trace
        def forward(self, x, scale, dy):
            return raf.layer_norm_dx(x, scale, dy, eps=self.eps)

    def gen_arg_func(batch_size, seq_length, hidden_size, eps):
        shape = [batch_size, seq_length, hidden_size]
        m_x, _ = randn(shape, device=device)
        m_scale, _ = randn((shape[2],), device=device)
        m_dy, _ = randn(shape, device=device)
        return [eps], [m_x, m_scale, m_dy]

    if space_dict is None:
        space_dict = {
            "batch_size": [1, 4, 8, 16, 32, 64],
            "seq_length": [128],
            "hidden_size": [768],
            "eps": [1e-5, 1e-12],
        }

    tune_op(
        sch_file,
        LayerNormDxModel,
        gen_arg_func,
        space_dict,
        device=device,
        only_extract_tasks=only_extract_tasks,
    )


def tune_take_dx(sch_file, space_dict=None, device="cuda", only_extract_tasks=False):
    """Tune take_dx with various shapes in transformer-based models.

    sch_file: str
        The tuning log.

    space_dict: Optional[Dict[str, List[Any]]]
        The target space (configs) of this op. If not present, the default space will be used.

    device: str
        The target device. Default cuda.

    only_extract_tasks: bool
        Whether to extract and print tasks only without actual tuning them.
    """

    class TakeDxModel(raf.Model):
        def build(self):
            pass

        @raf.model.trace
        def forward(self, x, dy, indices):
            return raf.take_dx(x, dy, indices, axis=0, mode="clip")

    def gen_arg_func(batch_size, seq_length, hidden_size, vocab_size):
        x_shape = (vocab_size, hidden_size)
        y_shape = (batch_size, seq_length, hidden_size)
        i_shape = (batch_size, seq_length)
        m_x, _ = randn(x_shape, device=device)
        m_dy, _ = randn_torch(y_shape, std=0.0, mean=1.0, requires_grad=False, device=device)
        m_indices, _ = randint(i_shape, low=0, high=y_shape[1], dtype="int32", device=device)
        return [], [m_x, m_dy, m_indices]

    if space_dict is None:
        space_dict = {
            "batch_size": [1, 4, 8, 16, 32, 64],
            "seq_length": [128],
            "hidden_size": [768],
            "vocab_size": [2, 512, 1024, 30522, 50257],
        }

    tune_op(
        sch_file,
        TakeDxModel,
        gen_arg_func,
        space_dict,
        device=device,
        only_extract_tasks=only_extract_tasks,
    )


def tune_fused_take_dx(sch_file, space_dict=None, device="cuda", only_extract_tasks=False):
    """Tune fused take_dx with various shapes in transformer-based models.

    sch_file: str
        The tuning log.

    space_dict: Optional[Dict[str, List[Any]]]
        The target space (configs) of this op. If not present, the default space will be used.

    device: str
        The target device. Default cuda.

    only_extract_tasks: bool
        Whether to extract and print tasks only without actual tuning them.
    """

    class TakeDxModel(raf.Model):
        def build(self):
            self.m = array(0.1, dtype="float32")

        @raf.model.trace
        def forward(self, x, dy, indices):
            mul = raf.multiply(self.m, x)
            out = raf.take_dx(x, dy, indices, axis=0, mode="clip")
            return raf.add(mul, out)

    def gen_arg_func(batch_size, seq_length, hidden_size, vocab_size):
        x_shape = (vocab_size, hidden_size)
        y_shape = (batch_size, seq_length, hidden_size)
        i_shape = (batch_size, seq_length)
        m_x, _ = randn(x_shape, device=device)
        m_dy, _ = randn_torch(y_shape, std=0.0, mean=1.0, requires_grad=False, device=device)
        m_indices, _ = randint(i_shape, low=0, high=y_shape[1], dtype="int32", device=device)
        return [], [m_x, m_dy, m_indices]

    if space_dict is None:
        space_dict = {
            "batch_size": [1, 4, 8, 16, 32, 64],
            "seq_length": [128],
            "hidden_size": [768],
            "vocab_size": [2, 512, 1024, 30522, 50257],
        }

    tune_op(
        sch_file,
        TakeDxModel,
        gen_arg_func,
        space_dict,
        device=device,
        fusion=True,
        only_extract_tasks=only_extract_tasks,
    )


def tune_all_ops(sch_file, device="cuda", only_extract_tasks=False):
    """Tune all listed ops.

    sch_file: str
        The tuning log.

    device: str
        The target device. Default cuda.

    only_extract_tasks: bool
        Whether to extract and print tasks only without actual tuning them.
    """
    for tune_func in [
        tune_fused_take_dx,
        tune_take_dx,
        tune_softmax_dx,
        tune_layer_norm,
        tune_layer_norm_dx,
    ]:
        tune_func(sch_file, device=device, only_extract_tasks=only_extract_tasks)
