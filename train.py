"""Main training script.

Examples
--------
python train.py --directory=test --problem=conv_train

Arguments
---------
--vgpu : int >= 1
    (debug) Number of virtual GPUs to create for testing. If 1, no virtual GPUs
    are created, and a mirrored strategy is created with all physical GPUs.
--vram : int
    Amount of vram to allocate per virtual GPU if testing.
--cpu : bool
    Whether to run on CPU instead of GPU.
--gpus : int[]
    Comma separated list of GPU indices to use on a multi-gpu system.
--initialize : bool
    If True, only initializes and does not actually train
--strategy : str
    Strategy to use.
--policy : str
    Policy to train.
--presets : str[]
    Commaa separated list of presets to apply.
(all other args)
    Passed as overrides to strategy/policy building.
"""

import os
import sys

from config import get_default, get_preset, ArgParser

args = ArgParser(sys.argv[1:])

# Finally ready to import tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import l2o
from gpu_setup import create_distribute

# Directory
directory = args.pop_get("--directory", default="weights")

# Distribute
vgpus = int(args.pop_get("--vgpu", default=1))
memory_limit = int(args.pop_get("--vram", default=12000))
gpus = args.pop_get("--gpus", default=None)
cpu = args.pop_get("--cpu", default=False, dtype=bool)
distribute = create_distribute(
    vgpus=vgpus, memory_limit=memory_limit, gpus=gpus, do_cpu=cpu)

# Pick up flags first
initialize_only = args.pop_check("--initialize")

# Default params
strategy = args.pop_get("--strategy", "repeat")
policy = args.pop_get("--policy", "rnnprop")
default = get_default(strategy=strategy, policy=policy)

# Build overrides
presets = args.pop_get("--presets", "")
overrides = []
if presets != "":
    for p in presets.split(','):
        overrides += get_preset(p)
overrides += args.to_overrides()

with distribute.scope():
    # Build strategy
    strategy = l2o.build(
        default, overrides, directory=directory, strict=True)

    # Train if not --initialize
    if not initialize_only:
        strategy.train()
