# Optimizer Amalgamation

[ICLR 2022] ["Optimizer Amalgamation"](https://openreview.net/pdf?id=VqzXzA9hjaX) by Tianshu Huang, Tianlong Chen, Sijia Liu, Shiyu Chang, Lisa Amini, Zhangyang Wang

## Setup and Basic Usage

### Basic Setup

1. Clone repository and submodules
```
git clone --recursive https://github.com/VITA-Group/OptimizerDistillation
```

2. Check dependencies:

| Library | Known Working | Known Not Working |
| - | - | - |
| tensorflow | 2.3.0, 2.4.1 | <= 2.2 |
| tensorflow_datasets | 3.1.0, 4.2.0 | n/a |
| pandas | 0.24.1, 1.2.4 | n/a |
| numpy | 1.18.5, 1.19.2 | >=1.20 |
| scipy | 1.4.1, 1.6.2 | n/a |

See [here](https://github.com/thetianshuhuang/l2o) for more dependency information.

### Load pre-trained optimizer

Pre-trained weights can be found in the ``releases" tab on github.
After downloading and unzipping, the optimizers can be loaded as an L2O framework extending tf.keras.optimizers.Optimizer:
```python
import tensorflow as tf
import l2o

# Folder is sorted as ```pre-trained/{distillation type}/{replicate #}
opt = l2o.load("pre-trained/choice-large/7")
# The following is True
isinstance(opt, tf.keras.optimizers.Optimizer)
```

Pre-trained weights for Mean distillation (small pool), Min-max distillation (small pool), Choice distillation (small pool), and Choice distillation (large pool) are included.
Each folder contains 8 replicates with varying performance.

### Included scripts

See the docstring for each script for a full list of arguments (debug, other testing args).

Common (technical) arguments:

| Arg | Type | Description |
| - | - | - |
| ```gpus``` | ```int[]``` | Comma separated list of GPUs (1) |
| ```cpu``` | ```bool``` | Whether to run on CPU instead of GPU |

(1) GPUs are specified by GPU index (i.e. as returned by ```gpustat```). If no ```--gpus``` are provided, all GPUs on the system are used. If no GPUs are installed, CPU will be used.

```evaluate.py```:

| Arg | Type | Description |
| - | - | - |
| ```problem``` | ```str``` | Problem to evaluate on. Can pass a comma separated list. |
| ```directory``` | ```str``` | Target directory to load from. Can pass a comma separated list. |
| ```repeat``` | ```int``` | Number of times to run evaluation. Default: 10 |

```train.py```: 

| Arg | Type | Description |
| - | - | - |
| ```strategy``` | ```str``` | Training strategy to use. |
| ```policy``` | ```str``` | Policy to train. |
| ```presets``` | ```str[]``` | Comma separated list of presets to apply. | 
| (all other args) | - | Passed as overrides to strategy/policy building. |

```baseline.py```:

| Arg | Type | Description |
| - | - | - |
| ```problem``` | ```str``` | Problem to evaluate on. Can pass a comma separated list. |
| ```optimizer``` | ```str``` | Name of optimizer to use. |

### Experiment folder structure

Experiment file path:
```
results/{policy_name}/{experiment_name}/{replicate_number}
```

Experiment file structure:
```
[root]
  > [checkpoint]
      > stage_{stage_0.0.0}.index
      > stage_{stage_0.0.0}.data-00000-of-00001
      > stage_{stage_0.1.0}.index
      > ....
  > [eval]
      > [{eval_problem_1}]
          > stage_{x.x.x}.npz
      > ....
  > [log]
      > stage_{stage_0.0.0}.npz
      > stage_{stage_0.1.0}.npz
      > ....
  > config.json
  > summary.csv
```

Key files:
- ```config.json```: experiment configuration (hyperparameters, technical details, etc)
- ```summary.csv```: log of training details (losses, training time, etc)

## Experiments

### Mean, min-max distillation

Training with min-max distillation, rnnprop as target, small pool, convolutional network for training:
```
python train.py \
    --presets=conv_train,adam,rmsprop,il_more \
    --strategy=curriculum \
    --policy=rnnprop \
    --directory=results/rnnprop/min-max/1
```

Evaluation:
```
python evaluate.py \
    --problem=conv_train \
    --directory=results/rnnprop/min-max/1 \
    --repeat=10
```

Min-max distillation is the default setting. To use mean distillation, add the ```reduce_mean``` preset.

### Choice distillation

Train the choice policy:
```
python train.py \
    --presets=conv_train,cl_fixed \
    --strategy=repeat \
    --policy=less_choice \
    --directory=results/less-choice/base/1
```

Train for the final distillation step:
```
python train.py \
    --presets=conv_train,less_choice,il_more \
    --strategy=curriculum \
    --policy=rnnprop \
    --directory=results/rnnprop/choice2/1
```

Evaluation:
```
python evaluate.py \
    --problem=conv_train \
    --directory=results/rnnprop/choice2/1 \
    --repeat=10
```

### Stability-Aware Optimizer Distillation

FGSM, PGD, Adaptive PGD, Gaussian, and Adaptive Gaussian perturbations are implemented.
| Perturbation | Description | Preset Name | Magnitude Parameter |
| - | - | - | - |
| FGSM | Fast Gradient Sign Method | ```fgsm``` | ```step_size``` |
| PGD | Projected Gradient Descent | ```pgd``` | ```magnitude``` |
| Adaptive PGD | Adaptive PGD / "Clipped" GD | ```cgd``` | ```magnitude``` |
| Random | Random Gaussian | ```gaussian``` | ```noise_stddev``` |
| Adaptive Random | Random Gaussian, Adaptive Magnitude | ```gaussian_rel``` | ```noise_stddev``` |

Modify the magnitude of noise by passing
```
--policy/perturbation/config/[Magnitude Parameter]=[Desired Magnitude].
```

For PGD variants, the number of adversarial attack steps can also be modified:
```
--policy/perturbation/config/steps=[Desired Steps]
```
