from clearml import Task
from clearml.automation import HyperParameterOptimizer, optuna
from clearml.automation import UniformParameterRange
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

task = Task.init('examples', 'HyperParameterOptimizer example')

controller = HyperParameterOptimizer(
    base_task_id='7d1582793fe44fa7832fcef582489f52',
    hyper_parameters=[
        UniformParameterRange('Args/beta', min_value=0., max_value=1., step_size=0.1),
    ],
    objective_metric_title='UV L1',
    objective_metric_series='val',
    objective_metric_sign='min',
    max_number_of_concurrent_tasks=1,
    optimizer_class=optuna.OptimizerOptuna,
    execution_queue='workers', time_limit_per_job=2,
    total_max_jobs=100,
    max_iteration_per_job=500
)

# This will automatically create and print the optimizer new task id
# for later use. if a Task was already created, it will use it.
controller.set_time_limit(in_minutes=60.)
controller.start_locally()
# we can create a pooling loop if we like
# while not an_optimizer.reached_time_limit():
#     top_exp = an_optimizer.get_top_experiments(top_k=3)
#     print(top_exp)
# wait until optimization completed or timed-out
controller.wait()
# make sure we stop all jobs
controller.stop()
