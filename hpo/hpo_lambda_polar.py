import os

import optuna
from clearml import Task
from clearml.automation import HyperParameterOptimizer, UniformParameterRange, UniformIntegerParameterRange
from clearml.automation.optuna import OptimizerOptuna

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

task = Task.init('DenseSeg/HPO', 'polar: lambda')

controller = HyperParameterOptimizer(
    base_task_id='b0f963ad6026410c8e9d628965d1de71',
    hyper_parameters=[
        # lambda
        UniformParameterRange('Args/bce', min_value=0.5, max_value=10., step_size=0.5),
        UniformParameterRange('Args/reg_uv', min_value=0.5, max_value=10., step_size=0.5),
        UniformParameterRange('Args/tv', min_value=0.5, max_value=10., step_size=0.5),
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
    total_max_jobs=150,
    pool_period_min=1,
)

controller.start_locally()
# wait until optimization completed or timed-out
controller.wait()
# make sure we stop all jobs
controller.stop()

# save the optuna optimizer results
task.upload_artifact('optuna_study.pkl', artifact_object=controller.get_optimizer()._study, auto_pickle=True)
