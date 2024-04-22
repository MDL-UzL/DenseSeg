import os

import optuna
from clearml import Task
from clearml.automation import HyperParameterOptimizer, UniformIntegerParameterRange
from clearml.automation.optuna import OptimizerOptuna

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

task = Task.init('DenseSeg/HPO', 'heatmap')

controller = HyperParameterOptimizer(
    base_task_id='2a0ab59eafc84c0eb64ec348bc5e19eb',
    hyper_parameters=[
        # lambda
        UniformIntegerParameterRange('Args/std', min_value=1, max_value=10, step_size=1),
        UniformIntegerParameterRange('Args/alpha', min_value=1, max_value=50, step_size=1),
    ],
    objective_metric_title='TRE [mm]',
    objective_metric_series='val',
    objective_metric_sign='min',
    max_number_of_concurrent_tasks=1,
    optimizer_class=OptimizerOptuna,
    save_top_k_tasks_only=10,
    sampler=optuna.samplers.TPESampler(),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20),
    time_limit_per_job=10,
    max_iteration_per_job=100,
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
