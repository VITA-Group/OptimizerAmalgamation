"""Hyperparameter grid search.

Grid Search results are saved in "gridsearch/{problem}/{policy}".

Arguments
---------
--vgpu : int >= 1
    (debug) Number of virtual GPUs to create for testing. If 1, no virtual GPUs
    are created, and a mirrored strategy is created with all physical GPUs.
--problem : str
    Training problem to use.
--repeat : int
    Number of times to run each training run.
"""

import os
import sys
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

import l2o
from config import ArgParser, get_eval_problem
from gpu_setup import create_distribute


args = ArgParser(sys.argv[1:])
vgpus = args.pop_get("--vgpu", default=1, dtype=int)
distribute = create_distribute(vgpus=vgpus)
problem = args.pop_get("--problem", "conv_train")
repeat = args.pop_get("--repeat", default=10, dtype=int)
grid = args.pop_get("--grid", default="0.001,0.01,0.1")
grid = [float(x) for x in grid.split(",")]

# Other Hyperparams Fixed
policies = {
    "adam": lambda lr: l2o.policies.AdamOptimizer(
        learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07),
    "rmsprop": lambda lr: l2o.policies.RMSPropOptimizer(
        learning_rate=lr, rho=0.9),
    "sgd": lambda lr: l2o.policies.SGDOptimizer(learning_rate=lr),
    "momentum": lambda lr: l2o.policies.MomentumOptimizer(
        learning_rate=lr, beta_1=0.9),
    "addsign": lambda lr: l2o.policies.AddSignOptimizer(
        learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-10),
    "powersign": lambda lr: l2o.policies.PowerSignOptimizer(
        learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-10)
}


def _make_gridsearch_entry(policy, constructor, lr):

    print("Gridsearch: {}/{}/{}".format(problem, policy, lr))
    results = []
    for i in range(repeat):
        print("Gridsearch Evaluation {}/{}".format(i + 1, repeat))
        with distribute.scope():
            _res = l2o.evaluate.evaluate_model(
                constructor(lr).as_optimizer(), **get_eval_problem(problem))
        results.append(_res)

    dst = "gridsearch/{}/{}".format(problem, policy)
    os.makedirs(dst, exist_ok=True)

    results = {k: np.stack([d[k] for d in results]) for k in results[0]}
    np.savez(os.path.join(dst, str(lr)), **results)


print("Grid: {}".format(grid))
for policy, constructor in policies.items():
    for lr in grid:
        _make_gridsearch_entry(policy, constructor, lr)
