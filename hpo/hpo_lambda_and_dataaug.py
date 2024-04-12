import os

from clearml import Task
from clearml.automation import HyperParameterOptimizer, UniformParameterRange, UniformIntegerParameterRange
from clearml.automation.optuna import OptimizerOptuna
import optuna

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

task = Task.init('DenseSeg/HPO', 'lambda and data augmentation')

controller = HyperParameterOptimizer(
    base_task_id='9ed165baa5f746579bd72d9d2522ff20',
    hyper_parameters=[
        # lambda
        UniformParameterRange('Args/bce', min_value=0.5, max_value=10., step_size=0.5),
        UniformParameterRange('Args/reg_uv', min_value=0.5, max_value=10., step_size=0.5),
        UniformParameterRange('Args/tv', min_value=0.5, max_value=10., step_size=0.5),

        # data augmentation
        UniformIntegerParameterRange('Args/rotate', min_value=0, max_value=25, step_size=5),
        UniformParameterRange('Args/translate', min_value=0, max_value=0.2, step_size=0.05),
        UniformParameterRange('Args/scale', min_value=0, max_value=0.3, step_size=0.05),

    ],
    objective_metric_title=['UV L1', 'Landmark UV Loss', 'Dice'],
    objective_metric_series=['val', 'val', 'val'],
    objective_metric_sign=['min', 'min', 'max'],
    max_number_of_concurrent_tasks=1,
    optimizer_class=OptimizerOptuna,
    save_top_k_tasks_only=10,
    sampler=optuna.samplers.TPESampler(),
    time_limit_per_job=20,
    max_iteration_per_job=500,
    total_max_jobs=150

)

controller.start_locally()
# wait until optimization completed or timed-out
controller.wait()
# make sure we stop all jobs
controller.stop()
