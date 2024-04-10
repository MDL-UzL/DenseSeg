import os

from clearml import Task
from clearml.automation import HyperParameterOptimizer, UniformParameterRange, GridSearch

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

task = Task.init('DenseSeg/HPO', 'Smooth L1 beta')

controller = HyperParameterOptimizer(
    base_task_id='df46ff8a0b694305824e429447027a84',
    hyper_parameters=[
        UniformParameterRange('Args/beta', min_value=0., max_value=1., step_size=0.2),
    ],
    objective_metric_title='UV L1',
    objective_metric_series='val',
    objective_metric_sign='min',
    max_number_of_concurrent_tasks=1,
    optimizer_class=GridSearch,
    save_top_k_tasks_only=10
)

controller.start_locally()
# wait until optimization completed or timed-out
controller.wait()
# make sure we stop all jobs
controller.stop()
