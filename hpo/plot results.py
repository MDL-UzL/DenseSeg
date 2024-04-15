from clearml import Task
import optuna.visualization.matplotlib as vis
import torch

task = Task.get_task(task_id='e334679396144792b0db6476b3028277')
study = task.artifacts['optuna_study.pkl'].get()

pareto_front = study.best_trials
obj_names = ['UV L1', 'Landmark UV Loss', 'Dice']

for i, objective in enumerate(obj_names):
    best_idx = torch.argmin(torch.tensor([t.values[i] for t in pareto_front]))
    best_trial = pareto_front[best_idx]
    print(f'Best trial for {objective}:')
    for name, val in zip(obj_names, best_trial.values):
        print(f'{name}: {val}')
    print(f'Trial params: {best_trial.params}\n')


vis.plot_pareto_front(study, targets=lambda t: (t.values[0], t.values[1], t.values[2]),
                      target_names=['UV L1', 'Landmark UV Loss', 'Dice'])

