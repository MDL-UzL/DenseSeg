from clearml import Task
import optuna.visualization.matplotlib as vis
import torch

uv_mode = 'polar'
print('UV mode:', uv_mode)
task = Task.get_task(task_id={'cartesian': 'e334679396144792b0db6476b3028277',
                              'polar': '37c1aa66f3944e8492d51ee2ee419d13'}[uv_mode])
study = task.artifacts['optuna_study.pkl'].get()

pareto_front = study.best_trials
obj_names = ['UV L1', 'Landmark UV Loss', 'Dice']
direction = ['minimize', 'minimize', 'maximize']

for i, objective in enumerate(obj_names):
    if direction[i] == 'minimize':
        best_idx = torch.argmin(torch.tensor([t.values[i] for t in pareto_front]))
    else:
        best_idx = torch.argmax(torch.tensor([t.values[i] for t in pareto_front]))
    best_trial = pareto_front[best_idx]
    print(f'Best trial for {objective}:')
    for name, val in zip(obj_names, best_trial.values):
        print(f'{name}: {val}')
    print(f'Trial params: {best_trial.params}\n')


vis.plot_pareto_front(study, targets=lambda t: (t.values[0], t.values[1], t.values[2]),
                      target_names=['UV L1', 'Landmark UV Loss', 'Dice'])

