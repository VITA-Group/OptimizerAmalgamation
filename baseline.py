"""Evaluate Baseline.

Baseline results are saved in the ```./baseline``` folder.

Examples
--------
python baseline.py --problem=conv_train --optimizer=adam

Arguments
---------
--vgpu : int >= 1
    (debug) Number of virtual GPUs to create for testing. If 1, no virtual GPUs
    are created, and a mirrored strategy is created with all physical GPUs.
--cpu : bool
    Whether to run on CPU instead of GPU.
--gpus : int[]
    Comma separated list of GPU indices to use on a multi-gpu system.
--keras : bool
    Whether to use keras versions of each optimizer or manually coded version.
--problem : str
    Training problem to use.
--optimizer : str
    Name of optimizer to use.
--repeat : int
    Number of times to run evaluation.
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
cpu = args.pop_get("--cpu", default=False, dtype=bool)
gpus = args.pop_get("--gpus", default=None)
use_keras = args.pop_get("--keras", default=True, dtype=bool)
distribute = create_distribute(vgpus=vgpus, do_cpu=cpu, gpus=gpus)

problems = args.pop_get("--problem", "conv_train")

target = args.pop_get("--optimizer", "adam")
target_cfg = {
    "adam": {
        "class_name": "Adam",
        "config": {"learning_rate": 0.005, "beta_1": 0.9, "beta_2": 0.999}
    },
    "rmsprop": {
        "class_name": "RMSProp",
        "config": {"learning_rate": 0.005, "rho": 0.9}
    },
    "sgd": {
        "class_name": "SGD",
        "config": {"learning_rate": 0.2}
    },
    "momentum": {
        "class_name": "SGD",
        "config": {"learning_rate": 0.5, "momentum": 0.9}
    },
    "momentum_custom": {
        "class_name": "Momentum",
        "config": {"learning_rate": 0.5, "beta_1": 0.9}
    },
    "addsign": {
        "class_name": "AddSign",
        "config": {"learning_rate": 0.1, "beta_1": 0.9, "beta_2": 0.999}
    },
    "powersign": {
        "class_name": "PowerSign",
        "config": {"learning_rate": 0.1, "beta_1": 0.9, "beta_2": 0.999}
    },
    "adam_deep": {
        "class_name": "Adam",
        "config": {"learning_rate": 0.001, "beta_1": 0.9, "beta_2": 0.999}
    },
    "rmsprop_deep": {
        "class_name": "RMSProp",
        "config": {"learning_rate": 0.0005, "rho": 0.9}
    },
    "sgd_deep": {
        "class_name": "SGD",
        "config": {"learning_rate": 0.2}
    },
    "momentum_deep": {
        "class_name": "Momentum",
        "config": {"learning_rate": 0.2, "beta_1": 0.9}
    },
    "addsign_deep": {
        "class_name": "AddSign",
        "config": {"learning_rate": 0.05, "beta_1": 0.9, "beta_2": 0.999}
    },
    "powersign_deep": {
        "class_name": "PowerSign",
        "config": {"learning_rate": 0.05, "beta_1": 0.9, "beta_2": 0.999}
    },

}[target]

repeat = args.pop_get("--repeat", default=10, dtype=10)
problems = problems.split(",")

for problem in problems:
    kwargs = get_eval_problem(problem)
    if "steps" in kwargs:
        evaluator = l2o.evaluate.evaluate_function
    else:
        evaluator = l2o.evaluate.evaluate_model

    with distribute.scope():
        results = []
        for i in range(repeat):
            print("Evaluation Training {}/{}".format(i + 1, repeat))

            if use_keras:
                opt = tf.keras.optimizers.get(target_cfg)
            else:
                pol = l2o.deserialize.policy(target_cfg)
                opt = pol.architecture(pol)
            results.append(evaluator(opt, **kwargs))
        results = {k: np.stack([d[k] for d in results]) for k in results[0]}

        os.makedirs(os.path.join("baseline", target), exist_ok=True)
        np.savez(os.path.join("baseline", target, problem), **results)
