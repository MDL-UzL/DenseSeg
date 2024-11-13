import os

import optuna
from clearml import Task
from clearml.automation import HyperParameterOptimizer, UniformParameterRange
from clearml.automation.optuna import OptimizerOptuna
import torch

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

task = Task.init('DenseSeg/HPO', 'heatmap_segmentation_lambda')

controller = HyperParameterOptimizer(
    base_task_id='4ae03bca6e0c4371852d32125539267b',
    hyper_parameters=[
        UniformParameterRange('Args/lambda_loss', min_value=0.1, max_value=0.9, step_size=0.01),
    ],
    objective_metric_title=['TRE [mm]', 'Dice'],
    objective_metric_series=['val', 'val'],
    objective_metric_sign=['min', 'max'],
    max_number_of_concurrent_tasks=1,
    optimizer_class=OptimizerOptuna,
    save_top_k_tasks_only=10,
    sampler=optuna.samplers.GridSampler({
        'Args/lambda_loss': torch.arange(0.1, 0.9, 0.01).tolist()
    }),
    time_limit_per_job=12,
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
